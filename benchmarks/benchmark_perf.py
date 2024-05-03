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
parser.add_argument("--model-name", type=str, default="state-spaces/mamba-1.4b")
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--promptlen", type=int, default=2048)
parser.add_argument("--genlen", type=int, default=128)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--topp", type=float, default=0.9)
parser.add_argument("--minp", type=float, default=0.0)
parser.add_argument("--repetition-penalty", type=float, default=1.2)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--dtype", type=str, default='float16')
parser.add_argument("--perf_log", type=str, default='perf_log')
args = parser.parse_args()

# repeats = 3
device = "cuda"
if args.dtype.lower() == 'float16':
    args.dtype = torch.float16
elif args.dtype.lower() == 'bfloat16':
    args.dtype = torch.bfloat16
elif args.dtype.lower() == 'float32':
    args.dtype = torch.float32


from mamba_ssm.utils.amd import hip_optimize_linear
# import torch.nn as nn
# def align_weight_linear_layers(model):
#     """
#     make all Linear layers' weights to be contiguous over K dim
#     """
#     for name, module in model.named_children():
#         if isinstance(module, nn.Linear) and name in ['out_proj', 'x_proj']:
#             w = module.weight.t()
#             w = w.contiguous()
#             module.weight = torch.nn.Parameter(w.t())
#             # print(f"{name}: {module}.weight.t().contiguous().t()")
#         elif isinstance(module, nn.Module):
#             align_weight_linear_layers(module)
#     return model


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

# align_weight = False
# if align_weight:
#     align_weight_linear_layers(model)
#     # hip_optimize_linear(model)

quick_run = False
if quick_run:
    # test run
    cg = False
    torch.random.manual_seed(0)
    if args.prompt is None:
        input_ids = torch.randint(1, 1000, (args.batch, args.promptlen), dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    else:
        tokens = tokenizer(args.prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(device=device)
        attn_mask = tokens.attention_mask.to(device=device)
    max_length = input_ids.shape[1] + args.genlen
    if is_mamba:
        fn = lambda: model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=cg,
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
    out = fn()
    early_exit = True

    do_profile = True
    # profile
    if do_profile:
        from torch.profiler import profile, ProfilerActivity
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes=True,
                        with_flops=True,
                        with_modules=True,
                        profile_memory=True,
                        with_stack=True) as prof:
        # with profile(activities=[ProfilerActivity.CUDA],
        #                 record_shapes=False,
        #                 with_flops=True,
        #                 with_modules=True,
        #                 profile_memory=True,
        #                 with_stack=False) as prof:
                out = fn()
        fname_prefix = f"benchmark_perf"
        postfix = f"B{input_ids.shape[0]}.N{input_ids.shape[1]}.L{out.sequences[-1].shape[0]}.{args.dtype}"
        print(f"input = {input_ids.shape}")
        print(f"output = {out.sequences[-1].shape}")
        print(f"profile to {fname_prefix}-{postfix}-trace.json")
        print(postfix)
        prof.export_chrome_trace(f"{fname_prefix}-{postfix}-trace.json")
        perf_table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=16)
        print(perf_table)


        # Split rows based on newline character
        perf_table = prof.key_averages().table(sort_by="self_cuda_time_total")
        rows = perf_table.split("\n")

    if early_exit:
        import sys;sys.exit(1)

def run_benchmark(args, seed=0, B=[1,2,4,8,16,32,64,128], N=None, n_warmups=1, cg=True):
    if B is None:
        B = [args.batch]
    if N is None:
        N = [args.promptlen]
    perf_log = args.perf_log 
    for promptlen in N:
        for batch in B:
            args.batch = batch
            args.promptlen = promptlen
            print(f"## run_benchmark with batch_size={batch}, args={args}")
            torch.random.manual_seed(seed)
            input_ids = torch.randint(1, 1000, (batch, args.promptlen), dtype=torch.long, device="cuda")
            attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")
            max_length = input_ids.shape[1] + args.genlen

            # warm-up
            if is_mamba:
                fn = lambda: model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    cg=cg,
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
                    cg=cg,
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
                print(f"profile to {fname_prefix}-{postfix}-trace.json")
                print(postfix)
                prof.export_chrome_trace(f"{fname_prefix}-{postfix}-trace.json")
                perf_table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=16)
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

            # repeats = 3
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


import triton
import triton.ops.matmul as triton_matmul
import torch.nn as nn
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[
            (32*2048, 8192, 2048), # in_proj
            (32*2048, 160, 4096), # x_proj
            (32*2048, 4096, 128), # dt_proj
            (32*2048, 2048, 4096), # out_proj
            # (131072, 4096, 2048)
        ],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['rocblas', 'linear', 'linear-a.t', 'rocblas-a.t-b.t', 'rocblas-a.t', 'rocblas-b.t'], #, "triton_matmul"],
        # Label name for the lines
        line_names=["rocBLAS", 'torch.linear', 'torch.linear-a.t', 'rocblas-a.t-b.t', 'rocblas-a.t', 'rocblas-b.t'], #, "triton_matmul"],
        # Line styles
        # styles=[('green', '-'), ('blue', '-'), ('red', '-'), ('red', '--')],
        styles=[('k', '--'), ('k', '-'), ('red', '-'), ('red', '--'), ('g', '-'), ('g', '--'), ('b', '-'), ('b', '--'),
                ('m', '-'), ('m', '--'), ('c', '-'), ('y', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name=f"gemm_perf_{args.dtype}",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark(M, N, K, provider, dtype=torch.float16):
    a = torch.randn((M, K), device='cuda', dtype=dtype)
    b = torch.randn((K, N), device='cuda', dtype=dtype)
    # b = torch.randn((N, K), device='cuda', dtype=dtype)
    # b = b.t()
    print(f"## {provider} benchmark {M}, {N}, {K} ##")
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'linear-a.t':
        a = torch.randn((K, M), device='cuda', dtype=dtype)
        a = a.t()
        out_proj = nn.Linear(K, N, device='cuda', dtype=dtype)
        # w = out_proj.weight.t()
        # w = w.contiguous()
        # out_proj.weight =  torch.nn.Parameter(w.t())
        fn = lambda: out_proj(a)
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    elif provider == 'linear':
        out_proj = nn.Linear(K, N, device='cuda', dtype=dtype)
        fn = lambda: out_proj(a)
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    elif provider == 'rocblas':
        fn = lambda: torch.matmul(a, b)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(), quantiles=quantiles)
    elif provider == 'rocblas-a.t':
        a = torch.randn((K, M), device='cuda', dtype=dtype)
        a = a.t()
        fn = lambda: torch.matmul(a, b)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(), quantiles=quantiles)
    elif provider == 'rocblas-b.t':
        b = torch.randn((N, K), device='cuda', dtype=dtype)
        b = b.t()
        fn = lambda: torch.matmul(a, b)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(), quantiles=quantiles)
    elif provider == 'rocblas-a.t-b.t':
        a = torch.randn((K, M), device='cuda', dtype=dtype)
        b = torch.randn((N, K), device='cuda', dtype=dtype)
        a = a.t()
        b = b.t()
        fn = lambda: torch.matmul(a, b)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(), quantiles=quantiles)
    elif provider == 'triton' or provider == 'triton_matmul':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_matmul(a, b), quantiles=quantiles)
        fn = lambda: triton_matmul(a, b)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    # perf = lambda ms: ms
    print(f"latency: ms : {ms}")
    # return perf(ms), perf(max_ms), perf(min_ms)
    print(f"tflops: {perf(ms)}")

    c = fn()
    dsize = a.nelement() * a.element_size() + b.nelement() * b.element_size() + c.nelement() * c.element_size()
    mem_bw = lambda ms: (dsize * 1e-9) / (ms * 1e-3)
    print(f"{provider}:{M},{N},{K}")
    print(f"mem bw: GB/s: {mem_bw(ms)}")

    return perf(ms), perf(max_ms), perf(min_ms)


gemm_test = True
if gemm_test:
    print(f"Running matmul benchmark!!")
    benchmark.run(show_plots=True, print_data=True, save_path='.', dtype=args.dtype)

# run_benchmark(args, B=[1,2,4,8,16,32,64,128], N=[16,32,64,128,256,512,1024,2048])
# run_benchmark(args, B=[1], N=[128, 256])
# run_benchmark(args, B=[64], N=[16,32,64,128,256,512,1024,2048])
# run_benchmark(args, B=[32], N=[2048], cg=True)
run_benchmark(args, B=[1,2,4,8,16,32,64,128], N=[2048], cg=True)
write2csv(args.perf_log)
