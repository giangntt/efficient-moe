from data_utils import get_loaders
import torch
import time
import gc
from tqdm import tqdm
from collections import defaultdict
import os
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

class BaseEvaluator:

    def __init__(self, model, tokenizer, model_st, dev, args):
        self.model_name = model.__class__.__name__
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.model_st = model_st
        self.dataset_name = args.dataset_name
        self.testloader = get_loaders(args.dataset_name, self.tokenizer, seed=args.seed, seqlen=self.model.seqlen, eval_mode=True)
        self.dev = dev
        self.args = args
        self.dtype = torch.bfloat16
        self.require_position_embeddings = None
        self.num_dense_layers = getattr(self.model.config, 'first_k_dense_replace', 0)
        self.total_layers = getattr(self.model.config, 'num_hidden_layers', None)
        self.outliers_info_path = os.path.join(self.args.save_path, "outliers_info")
        self.massive_experts_info_path = os.path.join(self.args.save_path, "massive_experts_info")
        self.layer_wise_save_path_of_massive = os.path.join(self.massive_experts_info_path, "layer_wise_analysis")
        self.expert_wise_save_path_of_massive = os.path.join(self.massive_experts_info_path, "expert_wise_analysis")
        self.super_experts_info_path = os.path.join(self.args.save_path, "super_experts_info")
        # self.super_experts_info_path = "/home/sora/llm/moe/output/OLMoE-1B-7B-0924-tinystories/super_experts_info"
        self.layer_wise_save_path_of_super = os.path.join(self.super_experts_info_path, "layer_wise_analysis")
        self.expert_wise_save_path_of_super = os.path.join(self.super_experts_info_path, "expert_wise_analysis")
        self.super_experts_report_path = os.path.join(self.super_experts_info_path, "super_experts_report")
        self.expert_frequency_info_path = os.path.join(self.args.save_path, "expert_frequency_info")
        self.expert_frequency_report_path = os.path.join(self.expert_frequency_info_path, "expert_frequency_report.json")
        self.prune_experts=None
        self.SE_list = []

    def _get_module(self, model, submodule_key):
        sub_tokens = submodule_key.split('.')
        cur_mod = model
        for s in sub_tokens:
            cur_mod = getattr(cur_mod, s)
        return cur_mod

    def _load_layer_weight(self, layer_idx):
        raise NotImplementedError("Subclasses should implement the evaluate_ppl method.")
    
    def _layer_wise_evaluate(self, require_position_embeddings, require_hook, require_frequency_hook=False):
        dev = self.dev
        dtype = self.dtype
        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False    

        layers = self.model.model.layers
        # load embed_tokens
        self.model.model.embed_tokens = self.model.model.embed_tokens.to_empty(device=dev)
        self.model.model.embed_tokens.weight.data.copy_(self.model_st['model.embed_tokens.weight'].to(device=dev, dtype=dtype))

        layers[0] = self._load_layer_weight(0)

        # Convert the whole text of evaluation dataset into batches of sequences.
        input_ids = self.testloader.input_ids  # (1, text_len)
        nsamples = input_ids.numel() // self.model.seqlen  # The tail is truncated.
        input_ids = input_ids[:, :nsamples * self.model.seqlen].view(nsamples, self.model.seqlen).to(dev)  # (nsamples, seqlen)
        batch_size = 1
        input_ids = [input_ids[i:i + batch_size] for i in range(0, nsamples, batch_size)]
        nbatches = len(input_ids)
        inps = torch.zeros(
            (nbatches, batch_size, self.model.seqlen, self.model.config.hidden_size), dtype=dtype, device=dev
        )
        inps = [0] * nbatches
        if not require_position_embeddings:
            cache = {'i': 0, 'attention_mask': None}
            class Catcher(torch.nn.Module):
                def __init__(self, module):
                    super().__init__()
                    self.module = module
                def forward(self, inp, **kwargs):
                    inps[cache['i']] = inp
                    cache['i'] += 1
                    cache['attention_mask'] = kwargs['attention_mask']
                    raise ValueError
        else:
            cache = {'i': 0, 'attention_mask': None}
            cache = {'i': 0, 'position_embeddings': None}
            class Catcher(torch.nn.Module):
                def __init__(self, module):
                    super().__init__()
                    self.module = module
                def forward(self, inp, **kwargs):
                    inps[cache['i']] = inp
                    cache['i'] += 1
                    cache['attention_mask'] = kwargs['attention_mask']
                    cache['position_embeddings'] = kwargs['position_embeddings']
                    raise ValueError

        layers[0] = Catcher(layers[0])

        for i in range(nbatches):
            batch = input_ids[i]
            try:
                self.model(batch)
            except ValueError:
                pass
        layers[0] = layers[0].module
        layers[0] = layers[0].cpu()

        self.model.model.embed_tokens = self.model.model.embed_tokens.cpu()
        del self.model.model.embed_tokens
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(0.1)
        outs = [0] * nbatches

        if not require_position_embeddings:
            attention_mask = cache['attention_mask']
        else:
            attention_mask = cache['attention_mask']
            position_embeddings = cache['position_embeddings']

        # hook
        if require_hook:
            def experts_down_proj_hook(layer_index,layer_type, name):
                def hook(module, input, output):
                    input_shape = input[0].shape
                    # output_shape = output[0].shape
                    # print(f"name={name}")
                    # print(f"input_shape={output_shape}")
                    if input_shape[0] == 0:
                        inps_max = torch.tensor(0).to(input[0].device)
                        outps_max = torch.tensor(0).to(input[0].device)
                        inpsmax_index = torch.tensor(-1).to(input[0].device)
                        outpsmax_index= torch.tensor(-1).to(input[0].device)
                    else:
                        inps_max = input[0].abs().max()
                        inpsmax_index = input[0].abs().argmax()
                        outps_max = output[0].abs().max()
                        outpsmax_index = output[0].abs().argmax()
                    input_max_value[layer_index][layer_type].append(inps_max)
                    input_max_channel_index[layer_index][layer_type].append(inpsmax_index)
                    output_max_value[layer_index][layer_type].append(outps_max)
                    output_max_channel_index[layer_index][layer_type].append(outpsmax_index)
                return hook
            input_max_value = defaultdict(lambda: defaultdict(list))
            output_max_value = defaultdict(lambda: defaultdict(list))
            input_max_channel_index = defaultdict(lambda: defaultdict(list))
            output_max_channel_index = defaultdict(lambda: defaultdict(list))
            for name, module in self.model.named_modules():
                if "down" in name or "w2" in name:
                    layer_index = name.split('.')[2]
                    layer_type = name
                    module.register_forward_hook(experts_down_proj_hook(layer_index, layer_type, name))
                    
        if require_frequency_hook:
            self.expert_activation_counters = defaultdict(lambda: {'activation_count': 0, 'token_count': 0})
            
            def expert_frequency_hook(layer_type_name):
                def hook(module, input, output):
                    # input[0].shape[0] is the number of tokens routed to this expert in this batch
                    token_count_for_batch = input[0].shape[0]
                    if token_count_for_batch > 0:
                        # 'activation_count' logs how many forward passes (batches) included this expert
                        self.expert_activation_counters[layer_type_name]['activation_count'] += 1
                    # 'token_count' logs the total number of tokens processed by this expert
                    self.expert_activation_counters[layer_type_name]['token_count'] += token_count_for_batch
                return hook

            for name, module in self.model.named_modules():
                # We hook the same modules as the outlier profiler: the down-projection layer.
                # Its input shape reveals how many tokens were routed to it.
                # Added 'w3' for Mixtral down-projection.
                if "down" in name or "w2" in name or "w3" in name: 
                    # Ensure we're only hooking expert modules, not other down_projs
                    if "experts" in name or "shared_expert" in name:
                        module.register_forward_hook(expert_frequency_hook(name))
        # load layer and forward
        if not require_position_embeddings:
            for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
                layer = self._load_layer_weight(i)
                for j in range(nbatches):
                    with torch.no_grad(): 
                        outs[j] = layer(inps[j], attention_mask=attention_mask)[0]
                layers[i] = None
                del layer
                torch.cuda.empty_cache()
                time.sleep(0.1)
                inps, outs = outs, inps

        else:
            for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
                layer = self._load_layer_weight(i)
                for j in range(nbatches):
                    with torch.no_grad(): 
                        outs[j] = layer(inps[j], attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
                layers[i] = None
                del layer
                torch.cuda.empty_cache()
                time.sleep(0.1)
                inps, outs = outs, inps

        if require_hook:
            return input_max_value, output_max_value, input_max_channel_index, output_max_channel_index
        if require_frequency_hook:
            return self.expert_activation_counters
        else:
            return nbatches, inps, input_ids, use_cache

    def _vis_outliers_heatmap(self, save_path, num_dense_layers, total_layers):
        json_folder_path = save_path
        layer_input_max = {}
        layer_output_max = {}
        for file_name in os.listdir(json_folder_path):
            if file_name.endswith(".json"):
                with open(os.path.join(json_folder_path, file_name), 'r') as f:
                    data = json.load(f)
                    for entry in data:
                        layer_index = int(entry["layer_index"])
                        input_max = entry["input_max"]
                        if int(layer_index) < num_dense_layers:
                            continue
                        output_max = entry["output_max"]
                        if layer_index not in layer_input_max:
                            layer_input_max[layer_index] = []
                            layer_output_max[layer_index] = []
                        layer_input_max[layer_index].append(input_max)
                        layer_output_max[layer_index].append(output_max)

        layers = sorted(layer_input_max.keys())
        num_experts = max(len(layer_input_max[layer]) for layer in layers)

        heatmap_data_input = []
        for layer in layers:
            experts = layer_input_max[layer]
            heatmap_data_input.append(experts)

        heatmap_data_output = []
        for layer in layers:
            experts = layer_output_max[layer]
            heatmap_data_output.append(experts)

        figure_save_path = os.path.join(json_folder_path, "figure")
        os.makedirs(figure_save_path, exist_ok=True)
        plt.figure(figsize=(16, 6))
        sns.heatmap(heatmap_data_input, cmap="coolwarm", xticklabels=[f"{i}" for i in range(0, num_experts)], yticklabels=[f"{layer}" for layer in layers], vmax=1000)
        plt.title("Input Max Values Heatmap", fontsize=24)
        plt.xlabel("Expert", fontsize=24)
        plt.ylabel("Layer", fontsize=24)
        plt.xticks(ticks=np.arange(0, num_experts, 10), labels=[f"{i}" for i in range(0, num_experts, 10)], fontsize=12)
        plt.yticks(ticks=np.arange(0, len(layers), 10), labels=[f"{layers[i]}" for i in range(0, len(layers), 10)], fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_save_path, "outliers_heap_map_input.pdf"))

        plt.figure(figsize=(16, 6))
        sns.heatmap(heatmap_data_output, cmap="coolwarm", xticklabels=[f"{i}" for i in range(0, num_experts)], yticklabels=[f"{layer}" for layer in layers], vmax=1000)
        plt.title("Output Max Values Heatmap", fontsize=24)
        plt.xlabel("Expert", fontsize=24)
        plt.ylabel("Layer", fontsize=24)
        plt.xticks(ticks=np.arange(0, num_experts, 10), labels=[f"{i}" for i in range(0, num_experts, 10)], fontsize=12)
        plt.yticks(ticks=np.arange(0, len(layers), 10), labels=[f"{layers[i]}" for i in range(0, len(layers), 10)], fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_save_path, "outliers_heap_map_output.pdf"))

    def _vis_massive_experts_line_plot(self, save_path, num_dense_layers, expert_wise):
        for filename in os.listdir(save_path):
            if filename.endswith(".json"):
                name = filename.split('.')[-2]
                file_path = os.path.join(save_path, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                if expert_wise or int(data[0]['layer_index']) >= num_dense_layers:
                    X = []
                    Y = []
                    if expert_wise:
                        sorted_data = sorted(data, key=lambda x:int(x['layer_index']), reverse=False)
                        for layer_info in sorted_data:
                            X.append(int(layer_info['layer_index']))
                            Y.append(layer_info['output_max'])
                    else:
                        shared_experts_info = []
                        router_experts_info = []            
                        for expert_info in data:
                            if int(expert_info['layer_index']) >= num_dense_layers:
                                if 'shared_experts' in expert_info["layer_type"]:
                                    shared_experts_info.append(expert_info)
                                else:
                                    router_experts_info.append(expert_info)                    
                        sorted_data = sorted(router_experts_info, key=lambda x: int(x["layer_type"].split('.')[-2]), reverse=False)
                        for expert_info in sorted_data:
                            X.append(int(expert_info['layer_type'].split('.')[-2]))
                            Y.append(expert_info['output_max'])  
                        for expert_info in shared_experts_info:
                            X.append(X[-1] + 1)
                            Y.append(expert_info['output_max'])  

                    plt.figure(figsize=(10, 6))
                    plt.plot(X, Y, marker='o', linestyle='-', color='b')
                    plt.ylabel('Output Max', fontsize=24)
                    if expert_wise:                
                        plt.xlabel('Layer Index', fontsize=24)
                        plt.title(f'Output Max for {name}', fontsize=20)
                    else:
                        plt.xlabel('Expert Index', fontsize=24)
                        plt.title(f'Output Max for {name}', fontsize=20)                    
                    plt.grid(True)
                    plt.xticks(fontsize=12) 
                    plt.yticks(fontsize=12)
                    plt.tight_layout()
                    figure_save_path = os.path.join(save_path, "figure")
                    os.makedirs(figure_save_path, exist_ok=True)
                    plot_filename = os.path.join(figure_save_path, f"{name}_plot.pdf")
                    plt.savefig(plot_filename)
                    plt.close()

    def _calculate_kurtosis(self, data):
        output_max_values = [entry['output_max'] for entry in data if 'output_max' in entry]
        return kurtosis(output_max_values, fisher=True) 

    def _plot_meta_from_json(self, save_path, object, expert_wise):
        if not expert_wise:
            layers = []
            meta_values = []
            for filename in os.listdir(save_path):
                if filename.endswith(".json"):
                    file_path = os.path.join(save_path, filename)
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        meta_data = data.get('meta_data', {})
                        meta = meta_data.get(object, None)
                        layer_index = meta_data.get('layer_index', None)
                        layers.append(int(layer_index))
                        meta_values.append(meta)
            sorted_layers, sorted_meta_values = zip(*sorted(zip(layers, meta_values)))
            plt.figure(figsize=(10, 6))
            plt.plot(sorted_layers, sorted_meta_values, marker='o', linestyle='-', color='b')
            plt.title(f"{object} by Layer", fontsize=24)
            plt.xlabel("Layer Index", fontsize=24)
            plt.grid(True)
            max_layer = max(sorted_layers)
            plt.xticks(ticks=np.arange(0, max_layer + 1, 10), labels=[f"{i}" for i in np.arange(0, max_layer + 1, 10)], fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            figure_path = os.path.join(self.super_experts_report_path, "figure")
            os.makedirs(figure_path, exist_ok=True)  
            plt.savefig(os.path.join(figure_path, f"{object} by Layer.pdf"))
            plt.show()
        else:
            shared_experts = []
            meta_values_of_shared_experts = []
            router_experts = []
            meta_values_of_router_experts = []
            for filename in os.listdir(save_path):
                if filename.endswith(".json"):
                    file_path = os.path.join(save_path, filename)
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        meta_data = data.get('meta_data', {})
                        meta = meta_data.get(object, None)
                        expert_index = meta_data.get('expert_index', None)
                        if "shared_experts" in expert_index:
                            shared_experts.append(expert_index)
                            meta_values_of_shared_experts.append(meta)
                        else:
                            router_experts.append(int(expert_index))
                            meta_values_of_router_experts.append(meta)
            sorted_experts, sorted_meta_values = zip(*sorted(zip(router_experts, meta_values_of_router_experts)))
            sorted_experts = list(sorted_experts)
            sorted_meta_values = list(sorted_meta_values)
            for shared_expert in range(len(shared_experts)):
                sorted_experts.append(shared_experts[shared_expert])
                sorted_meta_values.append(meta_values_of_shared_experts[shared_expert])

            plt.figure(figsize=(10, 6))
            plt.plot(sorted_experts, sorted_meta_values, marker='o', linestyle='-', color='b')
            plt.title(f"{object} by Expert", fontsize=24)
            plt.xlabel("Expert Index", fontsize=24)
            plt.grid(True)
            max_expert = len(sorted_experts)
            plt.xticks(ticks=np.arange(0, max_expert + 1, 10), labels=[f"{i}" for i in np.arange(0, max_expert + 1, 10)], fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            figure_path = os.path.join(self.super_experts_report_path, "figure")
            os.makedirs(figure_path, exist_ok=True)  
            plt.savefig(os.path.join(figure_path, f"{object} by Expert.pdf"))
            plt.show()

    def _generate_SE_list(self):
        SE_json_path = os.path.join(self.super_experts_report_path, "Super Experts Report.json")
        with open(SE_json_path, 'r') as f:
            se_data = json.load(f)
        for entry in se_data:
            layer_index = entry['layer_index']
            expert_index = entry['expert_index']
            if expert_index == "shared_experts":
                expert_index = "-1"
            self.SE_list.append((int(layer_index),int(expert_index)))
        print(f"Super Experts: {self.SE_list}")
        


    def _outliers_info_to_json(self, save_path, input_max_value, output_max_value, input_max_channel_index, output_max_channel_index):
        for layer_index in input_max_value:
            outliers_file_path = os.path.join(save_path, f"layer_{layer_index}.json")
            if not os.path.exists(outliers_file_path):
                with open(outliers_file_path, 'w') as outliers_file:
                    json.dump([], outliers_file)
            for layer_type in input_max_value[layer_index]:
                input_max_value_stack = torch.stack(input_max_value[layer_index][layer_type])
                input_max_value[layer_index][layer_type], input_max_index = torch.max(input_max_value_stack, dim=0)
                output_max_value_stack = torch.stack(output_max_value[layer_index][layer_type])
                output_max_value[layer_index][layer_type], output_max_index = torch.max(output_max_value_stack, dim=0)
                layer_data = {
                    'layer_index':layer_index,
                    'layer_type':layer_type,
                    'input_max': input_max_value[layer_index][layer_type].item(),
                    # 'input_max_channel': input_max_channel_index[layer_index][layer_type][input_max_index].item(),
                    'output_max': output_max_value[layer_index][layer_type].item(),
                    'output_max_channel': output_max_channel_index[layer_index][layer_type][output_max_index].item(),
                }
                with open(outliers_file_path, 'r+') as outliers_file:
                    outliers_data = json.load(outliers_file)
                    outliers_data.append(layer_data)
                    outliers_file.seek(0)
                    json.dump(outliers_data, outliers_file, indent=4)

        # sort experts
        for filename in os.listdir(save_path):
            if filename.endswith(".json"):
                file_path = os.path.join(save_path, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                if "experts" in data[0]['layer_type']:
                    router_experts = []
                    shared_experts = []                    
                    for entry in data:
                        if 'mlp.experts' in entry['layer_type']:
                            router_experts.append(entry)
                        else:
                            shared_experts.append(entry)
                    experts_sorted = sorted(router_experts, key=lambda x: (int(x['layer_type'].split('.')[-2]), x['layer_type']))
                    experts_sorted.extend(shared_experts)
                    save_file_path = os.path.join(save_path, filename)
                    with open(save_file_path, 'w') as file:
                        json.dump(experts_sorted, file, indent=4)
        
    def _massive_experts_info_to_json(self, save_path, outliers_info_path):
        # layer-wise datas
        layer_wise_save_path = self.layer_wise_save_path_of_massive
        if not os.path.exists(layer_wise_save_path):
            os.makedirs(layer_wise_save_path)
        for filename in os.listdir(outliers_info_path):
            if filename.endswith(".json"):
                file_path = os.path.join(outliers_info_path, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                sorted_data = sorted(data, key=lambda x: x['output_max'], reverse=True)
                for idx, entry in enumerate(sorted_data, start=1):
                    entry['rank'] = idx
                save_file_path = os.path.join(layer_wise_save_path, filename)
                with open(save_file_path, 'w') as file:
                    json.dump(sorted_data, file, indent=4)

        # expert-wise datas
        expert_wise_save_path = self.expert_wise_save_path_of_massive
        if not os.path.exists(expert_wise_save_path):
            os.makedirs(expert_wise_save_path)
        experts_data = {}
        for filename in os.listdir(outliers_info_path):
            if filename.endswith(".json"):
                file_path = os.path.join(outliers_info_path, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    for entry in data:
                        if "experts" in entry['layer_type']:
                            if 'shared_experts' in entry['layer_type']:
                                expert_index = "shared_experts"
                            else:
                                expert_index = entry["layer_type"].split('.')[-2]
                            if expert_index not in experts_data:
                                experts_data[expert_index] = []
                            experts_data[expert_index].append(entry)

        for expert_index, expert_entries in experts_data.items():
            output_filename = f"expert_{expert_index}.json"
            output_path = os.path.join(expert_wise_save_path, output_filename)
            expert_entries = sorted(expert_entries, key=lambda x: x['output_max'], reverse=True)
            for idx, entry in enumerate(expert_entries, start=1):
                entry['rank'] = idx  
            with open(output_path, 'w') as output_file:
                json.dump(expert_entries, output_file, indent=4)

    def _super_experts_analysis(self, save_path, massive_experts_info_path, total_layers, expert_wise, include_layers=0.75):
            include_layers = round(total_layers * include_layers)
            for filename in os.listdir(massive_experts_info_path):
                if filename.endswith(".json"):
                    file_path = os.path.join(massive_experts_info_path, filename)
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                    new_data = []
                    for each in data:
                        if int(each['layer_index']) < include_layers:
                            new_data.append(each)
                    if new_data == []:
                        continue
                    kurt_value = self._calculate_kurtosis(new_data)
                    if not expert_wise:
                        meta_data = {
                            'include_layer': include_layers,
                            'output_max_kurtosis': kurt_value,
                            "layer_index": new_data[0]['layer_index'],
                            'max_output_max': new_data[0]['output_max'],
                            'kurtosis*max_output_max': kurt_value * new_data[0]['output_max']
                        }
                    else:
                        if 'shared_experts' in new_data[0]['layer_type']:
                            expert_index = "shared_experts"
                        else:
                            expert_index = new_data[0]["layer_type"].split('.')[-2]
                        meta_data = {
                            'include_layer': include_layers,
                            'output_max_kurtosis': kurt_value,
                            "expert_index": expert_index,
                            'max_output_max': new_data[0]['output_max'],
                            'kurtosis*max_output_max': kurt_value * new_data[0]['output_max']
                        }
                
                    updated_data = {
                        'meta_data': meta_data,
                        'data': new_data
                    }

                    with open(os.path.join(save_path, filename), 'w') as file:
                        json.dump(updated_data, file, indent=4)

    def _identify_super_experts_std(self, save_path, std_multiplier=3):
        results = []
        for filename in os.listdir(save_path):
            if filename.endswith(".json"):
                file_path = os.path.join(save_path, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    expert_index = data["meta_data"].get("expert_index", None)
                    for item in data.get("data", []):
                        output_max = item.get("output_max", None)
                        layer_index = item.get("layer_index", None)
                        if output_max is not None:
                            results.append({
                                "expert_index": expert_index,
                                "layer_index": layer_index,
                                "output_max": output_max
                            })

        output_max_values = [item['output_max'] for item in results]
        mean = np.mean(output_max_values)
        std_dev = np.std(output_max_values)
        threshold = mean + std_multiplier * std_dev

        Super_Experts = []
        for item in results:
            if item['output_max'] > threshold:
                Super_Experts.append({
                    'expert_index': item['expert_index'],
                    'layer_index': item['layer_index'],
                    'output_max': item['output_max']
                })
        Super_Experts.sort(key=lambda x: x['output_max'], reverse=True)
        for rank, item in enumerate(Super_Experts, start=1):
            item['rank'] = rank
        return Super_Experts

    def _identify_super_experts_quantile(self, save_path, quantile=99.5):
        results = []
        for filename in os.listdir(save_path):
            if filename.endswith(".json"):
                file_path = os.path.join(save_path, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    expert_index = data["meta_data"].get("expert_index", None)
                    for item in data.get("data", []):
                        output_max = item.get("output_max", None)
                        layer_index = item.get("layer_index", None)
                        if output_max is not None:
                            results.append({
                                "expert_index": expert_index,
                                "layer_index": layer_index,
                                "output_max": output_max
                            })

        output_max_values = [item['output_max'] for item in results]
        percentile = np.percentile(output_max_values, quantile)
        threshold = percentile

        Super_Experts = []
        for item in results:
            if item['output_max'] > threshold:
                Super_Experts.append({
                    'expert_index': item['expert_index'],
                    'layer_index': item['layer_index'],
                    'output_max': item['output_max']
                })
        Super_Experts.sort(key=lambda x: x['output_max'], reverse=True)
        for rank, item in enumerate(Super_Experts, start=1):
            item['rank'] = rank
        return Super_Experts

    def _identify_super_experts_aver(self, save_path, times=50):
        results = []
        for filename in os.listdir(save_path):
            if filename.endswith(".json"):
                file_path = os.path.join(save_path, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    expert_index = data["meta_data"].get("expert_index", None)
                    for item in data.get("data", []):
                        output_max = item.get("output_max", None)
                        layer_index = item.get("layer_index", None)
                        if output_max is not None:
                            results.append({
                                "expert_index": expert_index,
                                "layer_index": layer_index,
                                "output_max": output_max
                            })

        output_max_values = [item['output_max'] for item in results]
        average = np.mean(output_max_values)
        threshold = average * times

        Super_Experts = []
        for item in results:
            if item['output_max'] > threshold:
                Super_Experts.append({
                    'expert_index': item['expert_index'],
                    'layer_index': item['layer_index'],
                    'output_max': item['output_max']
                })
        Super_Experts.sort(key=lambda x: x['output_max'], reverse=True)
        for rank, item in enumerate(Super_Experts, start=1):
            item['rank'] = rank
        return Super_Experts

    def _identify_super_experts(self, save_path, quantile=99.5, times=10):
        results = []
        for filename in os.listdir(save_path):
            if filename.endswith(".json"):
                file_path = os.path.join(save_path, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    expert_index = data["meta_data"].get("expert_index", None)
                    for item in data.get("data", []):
                        output_max = item.get("output_max", None)
                        layer_index = item.get("layer_index", None)
                        if output_max is not None:
                            results.append({
                                "expert_index": expert_index,
                                "layer_index": layer_index,
                                "output_max": output_max
                            })

        output_max_values = [item['output_max'] for item in results]
        percentile = np.percentile(output_max_values, quantile)

        Super_Experts = []
        for item in results:
            if item['output_max'] > percentile and item['output_max'] > np.max(output_max_values) // times:
                Super_Experts.append({
                    'expert_index': item['expert_index'],
                    'layer_index': item['layer_index'],
                    'output_max': item['output_max']
                })
        Super_Experts.sort(key=lambda x: x['output_max'], reverse=True)
        for rank, item in enumerate(Super_Experts, start=1):
            item['rank'] = rank
        return Super_Experts
    def _save_expert_frequency_to_json(self, activation_counters):
        """Processes the raw activation counts and saves a structured JSON report."""
        os.makedirs(self.expert_frequency_info_path, exist_ok=True)
        
        all_experts_flat = []
        layer_wise_data = defaultdict(list)

        # Parse the module names to create a structured list
        for full_name, counts in activation_counters.items():
            parts = full_name.split('.')
            try:
                layer_index = int(parts[2])
                expert_id_num = None
                is_shared = False

                if 'shared_experts' in full_name or 'shared_expert' in full_name:
                    expert_id_num = -1 # Use -1 convention from your prune_args
                    is_shared = True
                elif 'experts' in full_name:
                    # Find the index of 'experts' and get the next part
                    expert_index_pos = parts.index('experts') + 1
                    expert_id_num = int(parts[expert_index_pos])

                if expert_id_num is None:
                    continue # Not a module we're tracking (e.g., non-expert down_proj)

                expert_info = {
                    'layer_index': layer_index,
                    'expert_index': expert_id_num,
                    'is_shared': is_shared,
                    'full_module_name': full_name,
                    'activation_count': counts['activation_count'], # How many batches it was used in
                    'token_count': counts['token_count'] # Total tokens processed
                }
                all_experts_flat.append(expert_info)
                layer_wise_data[layer_index].append(expert_info)

            except (ValueError, IndexError):
                print(f"Warning: Could not parse module name {full_name}")
                continue
        
        # --- Process Statistics ---
        if not all_experts_flat:
            print("No expert activations were recorded. Check hook logic.")
            return

        # Sort all experts globally by token count
        all_experts_flat.sort(key=lambda x: x['token_count'], reverse=True)
        
        total_tokens = sum(e['token_count'] for e in all_experts_flat)
        
        # Add token percentage to each expert
        for e in all_experts_flat:
            e['token_percentage'] = (e['token_count'] / total_tokens) * 100 if total_tokens > 0 else 0
        
        num_experts_to_report = min(20, len(all_experts_flat)) # Report top/bottom 20

        stats = {
            'total_tokens_processed_by_experts': total_tokens,
            'total_unique_experts_activated': len(all_experts_flat),
            'most_frequent_experts_by_token': all_experts_flat[:num_experts_to_report],
            'least_frequent_experts_by_token': all_experts_flat[-num_experts_to_report:][::-1], # Sliced and reversed
            'experts_with_zero_tokens': [e for e in all_experts_flat if e['token_count'] == 0]
        }

        # --- Process Layer-wise ---
        processed_layer_wise = []
        for layer_index in sorted(layer_wise_data.keys()):
            experts_in_layer = layer_wise_data[layer_index]
            # Sort experts within this layer by token count
            experts_in_layer.sort(key=lambda x: x['token_count'], reverse=True)
            processed_layer_wise.append({
                'layer_index': layer_index,
                'experts': experts_in_layer
            })

        # Combine stats and per-layer data into one report
        report = {
            'statistics_summary': stats,
            'layer_wise_breakdown': processed_layer_wise
        }

        # Save the report
        with open(self.expert_frequency_report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"Expert frequency report saved to {self.expert_frequency_report_path}")

    def expert_frequency_profiler(self):
        """Public method to run the expert frequency analysis."""
        os.makedirs(self.expert_frequency_info_path, exist_ok=True)
        print("Starting expert frequency profiling...")
        activation_counters = self._layer_wise_evaluate(
            self.require_position_embeddings, 
            require_hook=False, 
            require_frequency_hook=True
        )
        print("Frequency data collected. Generating report...")
        self._save_expert_frequency_to_json(activation_counters)
        return self.expert_frequency_info_path

    def evaluate_ppl(self, require_hook, prune_experts, prune_SE):
        self.prune_experts=prune_experts
        if prune_SE:
            self._generate_SE_list()
        dev = self.dev
        dtype = self.dtype
        nbatches, inps, input_ids, use_cache = self._layer_wise_evaluate(self.require_position_embeddings, require_hook, require_frequency_hook=False)
        self.model.eval()

        # load norm and lm_head
        self.model.model.norm = self.model.model.norm.to_empty(device=dev)
        self.model.model.norm.weight.data.copy_(self.model_st['model.norm.weight'].to(device=dev, dtype=dtype))
        self.model.lm_head = self.model.lm_head.to_empty(device=dev)
        self.model.lm_head.weight.data.copy_(self.model_st['lm_head.weight'].to(device=dev, dtype=dtype))
        nlls = []
        loss_fct = torch.nn.CrossEntropyLoss(reduction = "none")
        with torch.no_grad():
            for i in range(nbatches):
                hidden_states = inps[i]
                hidden_states = self.model.model.norm(hidden_states)
                lm_logits = self.model.lm_head(hidden_states)
                shift_logits = lm_logits[:, :-1, :]
                shift_labels = input_ids[i][:, 1:]
                loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
                neg_log_likelihood = loss.float().mean(dim=1)
                nlls.append(neg_log_likelihood)
            nlls_tensor = torch.cat(nlls)
            ppl = torch.exp(nlls_tensor.mean())
        self.model.config.use_cache = use_cache

        # release gpu memory
        self.model.model.norm.cpu()
        self.model.lm_head.cpu()
        del self.model.model.norm
        del self.model.lm_head
        torch.cuda.empty_cache()

        return ppl.item()

    def outliers_profiler(self, vis_outliers_heatmap, require_hook=True):
        os.makedirs(self.outliers_info_path, exist_ok=True)
        input_max_value, output_max_value, input_max_channel_index, output_max_channel_index = self._layer_wise_evaluate(self.require_position_embeddings, require_hook, require_frequency_hook=False)
        self._outliers_info_to_json(self.outliers_info_path, input_max_value, output_max_value, input_max_channel_index, output_max_channel_index)
        if vis_outliers_heatmap:
            self._vis_outliers_heatmap(self.outliers_info_path, self.num_dense_layers, self.total_layers)
        return self.outliers_info_path

    def massive_experts_profiler(self, vis_massive_experts_line_plot):
        os.makedirs(self.massive_experts_info_path, exist_ok=True)
        os.makedirs(self.expert_wise_save_path_of_massive, exist_ok=True)
        os.makedirs(self.layer_wise_save_path_of_massive, exist_ok=True)        
        self._massive_experts_info_to_json(self.massive_experts_info_path, self.outliers_info_path)
        if vis_massive_experts_line_plot:
            self._vis_massive_experts_line_plot(self.expert_wise_save_path_of_massive, self.num_dense_layers, expert_wise=True)            
            self._vis_massive_experts_line_plot(self.layer_wise_save_path_of_massive, self.num_dense_layers, expert_wise=False)
        return self.massive_experts_info_path
        
    def super_experts_profiler(self, include_layers, vis_super_experts_line_plot):
        os.makedirs(self.super_experts_info_path, exist_ok=True) 
        os.makedirs(self.expert_wise_save_path_of_super, exist_ok=True)
        os.makedirs(self.layer_wise_save_path_of_super, exist_ok=True) 
        os.makedirs(self.super_experts_report_path, exist_ok=True)
        self._super_experts_analysis(self.expert_wise_save_path_of_super, self.expert_wise_save_path_of_massive, self.total_layers, expert_wise=True, include_layers=include_layers)
        self._super_experts_analysis(self.layer_wise_save_path_of_super, self.layer_wise_save_path_of_massive, self.total_layers, expert_wise=False, include_layers=include_layers)       

        Super_Experts = self._identify_super_experts(self.expert_wise_save_path_of_super)
        with open(os.path.join(self.super_experts_report_path, 'Super Experts Report.json'), 'w') as f:
            json.dump(Super_Experts, f, indent=4)

        if vis_super_experts_line_plot:
            # for object in ['output_max_kurtosis', 'max_output_max', 'kurtosis*max_output_max']:      
            for object in ['max_output_max']:
                self._plot_meta_from_json(self.expert_wise_save_path_of_super, object, expert_wise=True)
                self._plot_meta_from_json(self.layer_wise_save_path_of_super, object, expert_wise=False) 


        return self.super_experts_info_path


class DeepSeekEvaluator(BaseEvaluator):
    def __init__(self, model, tokenizer, model_st, dev, args):
        super().__init__(model, tokenizer, model_st, dev, args) 
        self.require_position_embeddings = False

    def _load_layer_weight(self, layer_idx):
        if self.prune_experts is not None:
            prune_list_router = [f"model.layers.{layer}.mlp.experts.{expert}" for layer, expert in self.prune_experts if int(expert) != -1]   
        else:
            prune_list_router = []
        if self.SE_list:
            for layer, expert in self.SE_list:
                if int(expert) != -1:
                    prune_list_router.append(f"model.layers.{layer}.mlp.experts.{expert}")
        layer_key = f"model.layers.{layer_idx}"
        layer = self._get_module(self.model, layer_key)
        dev = self.dev
        dtype = self.dtype

        # initialize meta tensor of attention
        ## layernorm
        W = layer.input_layernorm.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.input_layernorm.weight'].to(device=dev, dtype=dtype))
        W = layer.post_attention_layernorm.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.post_attention_layernorm.weight'].to(device=dev, dtype=dtype))
        W = layer.self_attn.kv_a_layernorm.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.self_attn.kv_a_layernorm.weight'].to(device=dev, dtype=dtype))
        if hasattr(layer.self_attn, 'q_a_layernorm'):
            W = layer.self_attn.q_a_layernorm.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[layer_key + '.self_attn.q_a_layernorm.weight'].to(device=dev, dtype=dtype))

        ## mla
        if hasattr(layer.self_attn, 'q_b_proj'):
            W = layer.self_attn.q_a_proj.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[layer_key + '.self_attn.q_a_proj.weight'].to(device=dev, dtype=dtype))
            W = layer.self_attn.q_b_proj.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[layer_key + '.self_attn.q_b_proj.weight'].to(device=dev, dtype=dtype))
        else:
            W = layer.self_attn.q_proj.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[layer_key + '.self_attn.q_proj.weight'].to(device=dev, dtype=dtype))
        W = layer.self_attn.kv_a_proj_with_mqa.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.self_attn.kv_a_proj_with_mqa.weight'].to(device=dev, dtype=dtype))
        W = layer.self_attn.kv_b_proj.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.self_attn.kv_b_proj.weight'].to(device=dev, dtype=dtype))
        W = layer.self_attn.o_proj.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.self_attn.o_proj.weight'].to(device=dev, dtype=dtype))

        # initialize meta tensor of mlp
        if hasattr(layer.mlp, 'experts'):
            expert_num = len(layer.mlp.experts)
            ## experts
            for expert_idx in range(expert_num):
                expert_key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
                expert = self._get_module(self.model, expert_key)
                W = expert.up_proj.to_empty(device=dev).to(dtype=dtype)
                W.weight.data.copy_(self.model_st[expert_key + '.up_proj.weight'].to(device=dev, dtype=dtype))
                W = expert.gate_proj.to_empty(device=dev).to(dtype=dtype)
                W.weight.data.copy_(self.model_st[expert_key + '.gate_proj.weight'].to(device=dev, dtype=dtype))
                W = expert.down_proj.to_empty(device=dev).to(dtype=dtype)

                if expert_key in prune_list_router:
                    print(f"prune {expert_key}")
                    W.weight.data.copy_(torch.zeros_like(self.model_st[expert_key + '.down_proj.weight'].to(device=dev, dtype=dtype)))
                else:
                    W.weight.data.copy_(self.model_st[expert_key + '.down_proj.weight'].to(device=dev, dtype=dtype))

            ## router
            router_key = f"model.layers.{layer_idx}.mlp.gate"
            bias_key = f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"
            W = layer.mlp.gate.to_empty(device=dev).to(dtype=torch.float32)
            W.weight.data.copy_(self.model_st[router_key + '.weight'].to(device=dev, dtype=torch.float32))
            if hasattr(layer.mlp.gate, "e_score_correction_bias"):
                W.e_score_correction_bias.data.copy_(self.model_st[bias_key ].to(device=dev, dtype=torch.float32))
            W = layer.mlp.shared_experts.up_proj.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[layer_key + '.mlp.shared_experts.up_proj.weight'].to(device=dev, dtype=dtype))
            W = layer.mlp.shared_experts.gate_proj.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[layer_key + '.mlp.shared_experts.gate_proj.weight'].to(device=dev, dtype=dtype))
            W = layer.mlp.shared_experts.down_proj.to_empty(device=dev).to(dtype=dtype)
            if self.prune_experts is not None:
                prune_list_shared = [int(layer) for layer, expert in self.prune_experts if int(expert) == -1]
            else:
                prune_list_shared = []
            for LAYER, EXPERT in self.SE_list:
                if int(EXPERT) == -1:
                    prune_list_shared.append(int(LAYER))
            if layer_idx in prune_list_shared:
                print(f"prune model.layers.{layer_idx}.mlp.shared_experts")
                W.weight.data.copy_(torch.zeros_like(self.model_st[layer_key + '.mlp.shared_experts.down_proj.weight'].to(device=dev, dtype=dtype)))
            else:
                W.weight.data.copy_(self.model_st[layer_key + '.mlp.shared_experts.down_proj.weight'].to(device=dev, dtype=dtype))

        else:
            W = layer.mlp.up_proj.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[layer_key + '.mlp.up_proj.weight'].to(device=dev, dtype=dtype))
            W = layer.mlp.gate_proj.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[layer_key + '.mlp.gate_proj.weight'].to(device=dev, dtype=dtype))
            W = layer.mlp.down_proj.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[layer_key + '.mlp.down_proj.weight'].to(device=dev, dtype=dtype))

        return layer

class Qwen2MoeEvaluator(BaseEvaluator):
    def __init__(self, model, tokenizer, model_st, dev, args):
        super().__init__(model, tokenizer, model_st, dev, args) 
        self.require_position_embeddings = True  # Qwen2Moe uses rotary embeddings

    def _load_layer_weight(self, layer_idx):
        if self.prune_experts is not None:
            prune_list_router = [f"model.layers.{layer}.mlp.experts.{expert}" for layer, expert in self.prune_experts if int(expert) != -1]   
        else:
            prune_list_router = []
        if self.SE_list:
            for layer, expert in self.SE_list:
                if int(expert) != -1:
                    prune_list_router.append(f"model.layers.{layer}.mlp.experts.{expert}")
        
        layer_key = f"model.layers.{layer_idx}"
        layer = self._get_module(self.model, layer_key)
        dev = self.dev
        dtype = self.dtype

        # Initialize meta tensor of layernorm
        W = layer.input_layernorm.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.input_layernorm.weight'].to(device=dev, dtype=dtype))
        W = layer.post_attention_layernorm.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.post_attention_layernorm.weight'].to(device=dev, dtype=dtype))

        # Initialize meta tensor of attention (standard attention, not MLA)
        W = layer.self_attn.q_proj.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.self_attn.q_proj.weight'].to(device=dev, dtype=dtype))
        if hasattr(layer.self_attn.q_proj, 'bias') and layer.self_attn.q_proj.bias is not None:
            W.bias.data.copy_(self.model_st[layer_key + '.self_attn.q_proj.bias'].to(device=dev, dtype=dtype))
        
        W = layer.self_attn.k_proj.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.self_attn.k_proj.weight'].to(device=dev, dtype=dtype))
        if hasattr(layer.self_attn.k_proj, 'bias') and layer.self_attn.k_proj.bias is not None:
            W.bias.data.copy_(self.model_st[layer_key + '.self_attn.k_proj.bias'].to(device=dev, dtype=dtype))
        
        W = layer.self_attn.v_proj.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.self_attn.v_proj.weight'].to(device=dev, dtype=dtype))
        if hasattr(layer.self_attn.v_proj, 'bias') and layer.self_attn.v_proj.bias is not None:
            W.bias.data.copy_(self.model_st[layer_key + '.self_attn.v_proj.bias'].to(device=dev, dtype=dtype))
        
        W = layer.self_attn.o_proj.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.self_attn.o_proj.weight'].to(device=dev, dtype=dtype))

        # Initialize meta tensor of mlp
        if hasattr(layer.mlp, 'experts'):
            expert_num = len(layer.mlp.experts)
            
            # Load experts
            for expert_idx in range(expert_num):
                expert_key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
                expert = self._get_module(self.model, expert_key)
                
                W = expert.gate_proj.to_empty(device=dev).to(dtype=dtype)
                W.weight.data.copy_(self.model_st[expert_key + '.gate_proj.weight'].to(device=dev, dtype=dtype))
                
                W = expert.up_proj.to_empty(device=dev).to(dtype=dtype)
                W.weight.data.copy_(self.model_st[expert_key + '.up_proj.weight'].to(device=dev, dtype=dtype))
                
                W = expert.down_proj.to_empty(device=dev).to(dtype=dtype)
                if expert_key in prune_list_router:
                    print(f"prune {expert_key}")
                    W.weight.data.copy_(torch.zeros_like(self.model_st[expert_key + '.down_proj.weight'].to(device=dev, dtype=dtype)))
                else:
                    W.weight.data.copy_(self.model_st[expert_key + '.down_proj.weight'].to(device=dev, dtype=dtype))

            # Load router gate
            router_key = f"model.layers.{layer_idx}.mlp.gate"
            W = layer.mlp.gate.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[router_key + '.weight'].to(device=dev, dtype=dtype))

            # Load shared expert (singular)
            W = layer.mlp.shared_expert.gate_proj.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[layer_key + '.mlp.shared_expert.gate_proj.weight'].to(device=dev, dtype=dtype))
            
            W = layer.mlp.shared_expert.up_proj.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[layer_key + '.mlp.shared_expert.up_proj.weight'].to(device=dev, dtype=dtype))
            
            W = layer.mlp.shared_expert.down_proj.to_empty(device=dev).to(dtype=dtype)
            
            # Check if shared expert should be pruned
            if self.prune_experts is not None:
                prune_list_shared = [int(layer) for layer, expert in self.prune_experts if int(expert) == -1]
            else:
                prune_list_shared = []
            for LAYER, EXPERT in self.SE_list:
                if int(EXPERT) == -1:
                    prune_list_shared.append(int(LAYER))
            
            if layer_idx in prune_list_shared:
                print(f"prune model.layers.{layer_idx}.mlp.shared_expert")
                W.weight.data.copy_(torch.zeros_like(self.model_st[layer_key + '.mlp.shared_expert.down_proj.weight'].to(device=dev, dtype=dtype)))
            else:
                W.weight.data.copy_(self.model_st[layer_key + '.mlp.shared_expert.down_proj.weight'].to(device=dev, dtype=dtype))

            # Load shared expert gate
            W = layer.mlp.shared_expert_gate.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[layer_key + '.mlp.shared_expert_gate.weight'].to(device=dev, dtype=dtype))

        else:
            # Dense MLP (if no experts)
            W = layer.mlp.gate_proj.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[layer_key + '.mlp.gate_proj.weight'].to(device=dev, dtype=dtype))
            
            W = layer.mlp.up_proj.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[layer_key + '.mlp.up_proj.weight'].to(device=dev, dtype=dtype))
            
            W = layer.mlp.down_proj.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[layer_key + '.mlp.down_proj.weight'].to(device=dev, dtype=dtype))

        return layer
    
class Qwen3MoeEvaluator(BaseEvaluator):
    def __init__(self, model, tokenizer, model_st, dev, args):
        super().__init__(model, tokenizer, model_st, dev, args) 
        self.require_position_embeddings = True

    def _load_layer_weight(self, layer_idx):
        if self.prune_experts is not None:
            prune_list_router = [f"model.layers.{layer}.mlp.experts.{expert}" for layer, expert in self.prune_experts if int(expert) != -1]   
        else:
            prune_list_router = []
        if self.SE_list:
            for layer, expert in self.SE_list:
                if int(expert) != -1:
                    prune_list_router.append(f"model.layers.{layer}.mlp.experts.{expert}")
        # print(f"prune_list_router={prune_list_router}")

        layer_key = f"model.layers.{layer_idx}"
        layer = self._get_module(self.model, layer_key)
        dev = self.dev
        dtype = self.dtype

        # initialize meta tensor of attention
        ## layernorm
        W = layer.input_layernorm.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.input_layernorm.weight'].to(device=dev, dtype=dtype))
        W = layer.post_attention_layernorm.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.post_attention_layernorm.weight'].to(device=dev, dtype=dtype))
        W = layer.self_attn.k_norm.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.self_attn.k_norm.weight'].to(device=dev, dtype=dtype))
        W = layer.self_attn.q_norm.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.self_attn.q_norm.weight'].to(device=dev, dtype=dtype))


        ## mha
        W = layer.self_attn.q_proj.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.self_attn.q_proj.weight'].to(device=dev, dtype=dtype))
        W = layer.self_attn.k_proj.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.self_attn.k_proj.weight'].to(device=dev, dtype=dtype))
        W = layer.self_attn.v_proj.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.self_attn.v_proj.weight'].to(device=dev, dtype=dtype))
        W = layer.self_attn.o_proj.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.self_attn.o_proj.weight'].to(device=dev, dtype=dtype))
        ## rotary

        # initialize meta tensor of mlp
        if hasattr(layer.mlp, 'experts'):
            expert_num = len(layer.mlp.experts)
            ## experts
            for expert_idx in range(expert_num):
                expert_key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
                expert = self._get_module(self.model, expert_key)
                W = expert.up_proj.to_empty(device=dev).to(dtype=dtype)
                W.weight.data.copy_(self.model_st[expert_key + '.up_proj.weight'].to(device=dev, dtype=dtype))
                W = expert.gate_proj.to_empty(device=dev).to(dtype=dtype)
                W.weight.data.copy_(self.model_st[expert_key + '.gate_proj.weight'].to(device=dev, dtype=dtype))
                W = expert.down_proj.to_empty(device=dev).to(dtype=dtype)
                if expert_key in prune_list_router:
                    print(f"prune {expert_key}")
                    W.weight.data.copy_(torch.zeros_like(self.model_st[expert_key + '.down_proj.weight'].to(device=dev, dtype=dtype)))
                else:
                    W.weight.data.copy_(self.model_st[expert_key + '.down_proj.weight'].to(device=dev, dtype=dtype))
            ## router
            router_key = f"model.layers.{layer_idx}.mlp.gate"

            W = layer.mlp.gate.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[router_key + '.weight'].to(device=dev, dtype=dtype))
        else:
            W = layer.mlp.up_proj.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[layer_key + '.mlp.up_proj.weight'].to(device=dev, dtype=dtype))
            W = layer.mlp.gate_proj.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[layer_key + '.mlp.gate_proj.weight'].to(device=dev, dtype=dtype))
            W = layer.mlp.down_proj.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[layer_key + '.mlp.down_proj.weight'].to(device=dev, dtype=dtype))

        return layer


class MixtralEvaluator(BaseEvaluator):
    def __init__(self, model, tokenizer, model_st, dev, args):
        super().__init__(model, tokenizer, model_st, dev, args) 
        self.require_position_embeddings = True

    def _load_layer_weight(self, layer_idx):
        if self.prune_experts is not None:
            prune_list_router = [f"model.layers.{layer}.block_sparse_moe.experts.{expert}" for layer, expert in self.prune_experts if int(expert) != -1]   
        else:
            prune_list_router = []
        if self.SE_list:
            for layer, expert in self.SE_list:
                if int(expert) != -1:
                    prune_list_router.append(f"model.layers.{layer}.block_sparse_moe.experts.{expert}")
        # print(f"prune_list_router={prune_list_router}")
        layer_key = f"model.layers.{layer_idx}"
        layer = self._get_module(self.model, layer_key)
        dev = self.dev
        dtype = self.dtype

        # initialize meta tensor of attention
        ## layernorm
        W = layer.input_layernorm.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.input_layernorm.weight'].to(device=dev, dtype=dtype))
        W = layer.post_attention_layernorm.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.post_attention_layernorm.weight'].to(device=dev, dtype=dtype))

        ## mha
        W = layer.self_attn.q_proj.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.self_attn.q_proj.weight'].to(device=dev, dtype=dtype))
        W = layer.self_attn.k_proj.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.self_attn.k_proj.weight'].to(device=dev, dtype=dtype))
        W = layer.self_attn.v_proj.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.self_attn.v_proj.weight'].to(device=dev, dtype=dtype))
        W = layer.self_attn.o_proj.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[layer_key + '.self_attn.o_proj.weight'].to(device=dev, dtype=dtype))
        ## rotary
        # W = layer.self_attn.rotary_emb.to_empty(device=dev).to(device=dev, dtype=torch.float32)
        # W.inv_freq.data.copy_(model_st[layer_key + '.self_attn.rotary_emb.inv_freq'].to(device=dev, dtype=torch.float32))

        # initialize meta tensor of mlp
        expert_num = len(layer.block_sparse_moe.experts)
        ## experts
        for expert_idx in range(expert_num):
            expert_key = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}"
            expert = self._get_module(self.model, expert_key)
            W = expert.w1.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[expert_key + '.w1.weight'].to(device=dev, dtype=dtype))
            W = expert.w2.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[expert_key + '.w2.weight'].to(device=dev, dtype=dtype))
            W = expert.w3.to_empty(device=dev).to(dtype=dtype)

            if expert_key in prune_list_router:
                print(f"prune {expert_key}")
                W.weight.data.copy_(torch.zeros_like(self.model_st[expert_key + '.w3.weight'].to(device=dev, dtype=dtype)))
            else:
                W.weight.data.copy_(self.model_st[expert_key + '.w3.weight'].to(device=dev, dtype=dtype))
        ## router
        router_key = f"model.layers.{layer_idx}.block_sparse_moe.gate"
        W = layer.block_sparse_moe.gate.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[router_key + '.weight'].to(device=dev, dtype=dtype))
        return layer
    
class OlmoeEvaluator(BaseEvaluator):
    def __init__(self, model, tokenizer, model_st, dev, args):
        super().__init__(model, tokenizer, model_st, dev, args)
        # Olmoe uses rotary embeddings which are typically handled by passing
        # position IDs/embeddings through the layer's forward pass.
        self.require_position_embeddings = True

    def _load_layer_weight(self, layer_idx):
        # Set up the list of experts to prune (zero-out) if any are specified.
        if self.prune_experts is not None:
            prune_list_router = [f"model.layers.{layer}.mlp.experts.{expert}" for layer, expert in self.prune_experts if int(expert) != -1]
        else:
            prune_list_router = []
        if self.SE_list:
            for layer, expert in self.SE_list:
                if int(expert) != -1:
                    prune_list_router.append(f"model.layers.{layer}.mlp.experts.{expert}")

        layer_key = f"model.layers.{layer_idx}"
        layer = self._get_module(self.model, layer_key)
        dev = self.dev
        dtype = self.dtype

        # === Initialize LayerNorm weights ===
        # Olmoe uses OlmoeRMSNorm for its normalization layers.
        W = layer.input_layernorm.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[f"{layer_key}.input_layernorm.weight"].to(device=dev, dtype=dtype))

        W = layer.post_attention_layernorm.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[f"{layer_key}.post_attention_layernorm.weight"].to(device=dev, dtype=dtype))

        # === Initialize Self-Attention weights ===
        # This includes the main Q, K, V, O projections and the specific norms for Q and K.
        W = layer.self_attn.q_proj.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[f"{layer_key}.self_attn.q_proj.weight"].to(device=dev, dtype=dtype))

        W = layer.self_attn.k_proj.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[f"{layer_key}.self_attn.k_proj.weight"].to(device=dev, dtype=dtype))

        W = layer.self_attn.v_proj.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[f"{layer_key}.self_attn.v_proj.weight"].to(device=dev, dtype=dtype))

        W = layer.self_attn.o_proj.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[f"{layer_key}.self_attn.o_proj.weight"].to(device=dev, dtype=dtype))

        W = layer.self_attn.q_norm.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[f"{layer_key}.self_attn.q_norm.weight"].to(device=dev, dtype=dtype))

        W = layer.self_attn.k_norm.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[f"{layer_key}.self_attn.k_norm.weight"].to(device=dev, dtype=dtype))


        # === Initialize Sparse MoE Block (MLP) weights ===
        # This model has 64 experts per MoE block.
        expert_num = len(layer.mlp.experts)
        for expert_idx in range(expert_num):
            expert_key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            expert = self._get_module(self.model, expert_key)

            W = expert.gate_proj.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[f"{expert_key}.gate_proj.weight"].to(device=dev, dtype=dtype))

            W = expert.up_proj.to_empty(device=dev).to(dtype=dtype)
            W.weight.data.copy_(self.model_st[f"{expert_key}.up_proj.weight"].to(device=dev, dtype=dtype))

            W = expert.down_proj.to_empty(device=dev).to(dtype=dtype)
            # Apply pruning by zeroing out the down_proj weight if the expert is in the prune list.
            if expert_key in prune_list_router:
                print(f"prune {expert_key}")
                W.weight.data.copy_(torch.zeros_like(self.model_st[f"{expert_key}.down_proj.weight"].to(device=dev, dtype=dtype)))
            else:
                W.weight.data.copy_(self.model_st[f"{expert_key}.down_proj.weight"].to(device=dev, dtype=dtype))

        # Load the router/gate weight.
        router_key = f"model.layers.{layer_idx}.mlp.gate"
        W = layer.mlp.gate.to_empty(device=dev).to(dtype=dtype)
        W.weight.data.copy_(self.model_st[f"{router_key}.weight"].to(device=dev, dtype=dtype))

        return layer