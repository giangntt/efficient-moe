import argparse
import torch
import os
import transformers
from model_utils import get_model, get_model_safe_tensors
from eval_utils import DeepSeekEvaluator, Qwen3MoeEvaluator, MixtralEvaluator, OlmoeEvaluator, Qwen2MoeEvaluator

def parse_args():
    parser = argparse.ArgumentParser()

    # config
    parser.add_argument('--model_name', type=str, default='', help='')  
    parser.add_argument('--dataset_name', type=str, default='c4', help='')    
    parser.add_argument('--save_path', type=str, default='', help='') 
    parser.add_argument('--seed', type=float, default=83, help='') 

    # profile outliers
    parser.add_argument('--profile_outliers', action='store_true', default=False, help='profile max input/output outliers of down_proj') 
    parser.add_argument('--vis_outliers_heatmap', action='store_true', default=False, help='vis heatmap of max input/output outliers of down_proj across all layers and experts')    
    
    # profile massive experts  
    parser.add_argument('--profile_massive_experts', action='store_true', default=False, help='profile Massive Experts')
    parser.add_argument('--vis_massive_experts_line_plot', action='store_true', default=False, help='vis lineplot of max input/output outliers of down_proj across all layers for all experts')
    
    # profile super experts      
    parser.add_argument('--profile_super_experts', action='store_true', default=False, help='profile Super Experts')
    parser.add_argument('--vis_super_experts_line_plot', action='store_true', default=False, help='vis lineplot of max input/output outliers of down_proj across all layers for super experts')
    parser.add_argument('--include_layers', type=float, default=0.75, help='')     
    
    # eval
    parser.add_argument('--eval_ppl', action='store_true', default=False, help='') 
    parser.add_argument('--prune_experts', type=lambda x: [tuple(map(int, item.split(','))) for item in x.split(';')],default=None, help='Pass a list of tuples, each in the format "layer,expert_index" separated by semicolons, e.g., "1,-1;2,3". -1 indicates shared experts.')
    parser.add_argument('--prune_super_experts', action='store_true', default=False, help='') 
    parser.add_argument('--profile_expert_frequency', action='store_true', default=False, help='profile expert activation frequency and token counts')
    args = parser.parse_args()
    return args


def main(args):
    transformers.set_seed(args.seed)
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)

    # initialize model
    model, tokenizer = get_model(args.model_name)
    model_st = get_model_safe_tensors(args.model_name)    
    model_name = model.__class__.__name__
    print(model.config)
    dev = torch.device('cuda')
    model.eval()

    # initialize evaluator
    if model_name in ['DeepseekV3ForCausalLM', 'DeepseekV2ForCausalLM']:
        evaluator = DeepSeekEvaluator(model, tokenizer, model_st, dev, args)
    elif model_name in ['Qwen3MoeForCausalLM']:
        evaluator = Qwen3MoeEvaluator(model, tokenizer, model_st, dev, args)
    elif model_name in ["Qwen2MoeForCausalLM"]:
        evaluator = Qwen2MoeEvaluator(model, tokenizer, model_st, dev, args)
    elif model_name in ['MixtralForCausalLM']:
        evaluator = MixtralEvaluator(model, tokenizer, model_st, dev, args)
    elif model_name in ['OlmoeForCausalLM']:
        evaluator = OlmoeEvaluator(model, tokenizer, model_st, dev, args)
    else:
        raise NotImplementedError  

    # profilling super experts
    if args.profile_outliers or args.vis_outliers_heatmap:
        outliers_info_path = evaluator.outliers_profiler(args.vis_outliers_heatmap, require_hook=True)
        print(f"Outliers Info is in {outliers_info_path}.")

    if args.profile_massive_experts or args.vis_massive_experts_line_plot:
        if not os.path.exists(evaluator.outliers_info_path):
            outliers_info_path = evaluator.outliers_profiler(args.vis_outliers_heatmap, require_hook=True)
        massive_experts_info_path = evaluator.massive_experts_profiler(args.vis_massive_experts_line_plot)
        print(f"Massive Experts Info is in {massive_experts_info_path}")

    if args.profile_super_experts:
        if not os.path.exists(evaluator.outliers_info_path):
            outliers_info_path = evaluator.outliers_profiler(args.vis_outliers_heatmap, require_hook=True)
        if not os.path.exists(evaluator.massive_experts_info_path):
            massive_experts_info_path = evaluator.massive_experts_profiler(args.vis_massive_experts_line_plot)
        super_experts_info_path = evaluator.super_experts_profiler(args.include_layers, args.vis_super_experts_line_plot)
        print(f"Super Experts Info is in {super_experts_info_path}")
    if args.profile_expert_frequency:
        frequency_info_path = evaluator.expert_frequency_profiler()
        print(f"Expert Frequency Info is in {frequency_info_path}")
    # ppl test
    if args.eval_ppl:
        if args.prune_super_experts:
            if not os.path.exists(evaluator.super_experts_info_path):
                print("Please perform profiling of the Super Experts first.")
            else:
                ppl = evaluator.evaluate_ppl(require_hook=False, prune_experts=args.prune_experts, prune_SE=args.prune_super_experts)
                print(f"After pruning Super Experts, Model {model_name} Dataset {args.dataset_name} PPL: {ppl}.")               
        else:
            ppl = evaluator.evaluate_ppl(require_hook=False, prune_experts=args.prune_experts, prune_SE=args.prune_super_experts)
            print(f"Model {model_name} Dataset {args.dataset_name} PPL: {ppl}.")


if __name__ == '__main__':
    args = parse_args()
    main(args)
