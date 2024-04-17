# Copyright (c) 2023, Tri Dao, Albert Gu.

import argparse
import time
import json

import torch
import torch.nn.functional as F

from einops import rearrange

from transformers import AutoTokenizer, AutoModelForCausalLM

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--model-name", type=str, default="state-spaces/mamba-2.8b")
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--promptlen", type=int, default=2048)
parser.add_argument("--genlen", type=int, default=128)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--topp", type=float, default=0.9)
parser.add_argument("--minp", type=float, default=0.0)
parser.add_argument("--repetition-penalty", type=float, default=1.2)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--dtype", type=str, default='float32')
parser.add_argument("--perf_log", type=str, default='benchmark_perf')
args = parser.parse_args()

# repeats = 3
device = "cuda"
if args.dtype.lower() == 'float16':
    args.dtype = torch.float16
elif args.dtype.lower() == 'bfloat16':
    args.dtype = torch.bfloat16
elif args.dtype.lower() == 'float32':
    args.dtype = torch.float32

print(f"Loading model {args.model_name}")
is_mamba = args.model_name.startswith("state-spaces/mamba-")
if is_mamba:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained(args.model_name, device=device, dtype=args.dtype)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map={"": device}, torch_dtype=args.dtype)
model.eval()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
print(f"Model: {model}")
print("\n\n\n\n")

def run_benchmark(args, seed=0, B=[1,2,4,8,16,32,64,128], N=None, n_warmups=1):
    if B is None:
        B = [args.batch]
    if N is None:
        N = [args.promptlen]
    perf_log = args.perf_log 
    for promptlen in N:
        for batch in B:
            args.batch = batch
            args.promptlen = promptlen
            print(f" ## run_benchmark with batch_size={batch}, args={args}")
            torch.random.manual_seed(seed)
            input_ids = torch.randint(1, 1000, (batch, args.promptlen), dtype=torch.long, device="cuda")
            attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")
            max_length = input_ids.shape[1] + args.genlen

            # warm-up
            if is_mamba:
                fn = lambda: model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    cg=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    enable_timing=True,
                    temperature=args.temperature,
                    top_k=args.topk,
                    top_p=args.topp,
                    min_p=args.minp,
                    repetition_penalty=args.repetition_penalty,
                )
            else:
                fn = lambda: model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_length=max_length,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=args.temperature,
                    top_k=args.topk,
                    top_p=args.topp,
                    repetition_penalty=args.repetition_penalty,
                )
            for n in range(n_warmups):
                fn()

            # eval run
            if is_mamba:
                fn = lambda: model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    cg=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    enable_timing=True,
                    perf_log=perf_log,
                    temperature=args.temperature,
                    top_k=args.topk,
                    top_p=args.topp,
                    min_p=args.minp,
                    repetition_penalty=args.repetition_penalty,
                )
            else:
                fn = lambda: model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_length=max_length,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=args.temperature,
                    top_k=args.topk,
                    top_p=args.topp,
                    repetition_penalty=args.repetition_penalty,
                )
            # test_run = False
            # if test_run:
            #     out = fn()
            #     print(f"input = {input_ids.shape}")
            #     print(f"output = {out.sequences[-1].shape}")

            do_profile = True
            # profile
            if do_profile:
                from torch.profiler import profile, ProfilerActivity
                torch.cuda.synchronize()
                # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                #                 record_shapes=True,
                #                 with_flops=True,
                #                 with_modules=True,
                #                 profile_memory=True,
                #                 with_stack=True) as prof:
                with profile(activities=[ProfilerActivity.CUDA],
                                record_shapes=False,
                                with_flops=True,
                                with_modules=True,
                                profile_memory=True,
                                with_stack=False) as prof:
                        out = fn()
                fname_prefix = f"gen_mamba_simple"
                postfix = f"B{input_ids.shape[0]}.N{input_ids.shape[1]}.L{out.sequences[-1].shape[0]}.{args.dtype}"
                print(f"input = {input_ids.shape}")
                print(f"output = {out.sequences[-1].shape}")
                print(postfix)
                prof.export_chrome_trace(f"{fname_prefix}-postfix-trace.json")
                perf_table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
                print(perf_table)


                # Split rows based on newline character
                perf_table = prof.key_averages().table(sort_by="self_cuda_time_total")
                rows = perf_table.split("\n")
                # print(rows)

                # Split each row into cells based on whitespace (you might need to adjust the delimiter)
                # data = [row.split() for row in rows[2:]]  # Skip header rows if any
                # data = [row.split('  ') for row in rows[3:]]  # Skip header rows if any
                import re

                # Define the string with extra spaces and the tab character
                # Split the string using a regular expression
                data = [re.split(r"\s{2,}", row) for row in rows]
                def remove_empty_strings(data):
                    """Removes all empty strings ('') from a 2D list.

                    Args:
                        data: A 2D list containing strings.

                    Returns:
                        A new 2D list with empty strings removed.
                    """
                    return [[item for item in sublist if item] for sublist in data]
                def remove_lists_with_double_dash(data):
                    """Removes lists from a list of strings where any element starts with '--'.

                    Args:
                        data: A list of lists containing strings.

                    Returns:
                        A new list of lists with lists containing '--' elements removed.
                    """
                    return [sublist for sublist in data if not any(item.startswith('--') for item in sublist)]

                data = remove_lists_with_double_dash(data)
                data = remove_empty_strings(data)


                # Print the split list
                # print(data)
                # headers = ['Name', 'Self CPU %', 'Self CPU', 'CPU total %', 'CPU total'  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem    # of Calls]
                import csv
                # Open the CSV file for writing
                csv_fname = f"torch-profile-{postfix}.csv"
                with open(csv_fname, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)

                    # Write the data to the CSV file
                    writer.writerows(data)

                # print(out.sequences.tolist())
                # print(tokenizer.batch_decode(out.sequences.tolist()))

            # torch.cuda.synchronize()
            # start = time.time()
            # for _ in range(repeats):
            #     fn()
            # torch.cuda.synchronize()
            # print(f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}")
            # print(f"{args.model_name} prompt processing + decoding time: {(time.time() - start) / repeats * 1000:.0f}ms")

def write2csv(perf_log):
    import csv, json, os
    perf_file = perf_log+".json"
    if os.path.exists(perf_file):
        with open(perf_file, "r") as f:
            perf_str = f.read()
        perf_dicts = json.loads(perf_str)
    with open(perf_log + ".csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(perf_dicts.keys())
        transposed_data = zip(*perf_dicts.values())
        for row in transposed_data:
            writer.writerow(row)

# run_benchmark(args, B=[1,2,4,8,16,32,64,128], N=[16,32,64,128,256,512,1024,2048])
# run_benchmark(args, B=[1], N=[128, 256])
# run_benchmark(args, B=[64], N=[16,32,64,128,256,512,1024,2048])
run_benchmark(args, B=[1,2,4,8,16,32,64,128], N=[2048])
write2csv(args.perf_log)
