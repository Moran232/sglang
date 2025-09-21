## benchmark data说明

- 用于测试不同接受率下的投机性能, 数据集从LongBench的21个子集中,分别选取10条, 经去重和筛选, 每条数据截取了前1024个token, 保证在deepseek r1 tokenizer下的输入token=1024, 输出token=1024;

- 五组benchmark_40_x.jsonl, 接受率从低到高, 大致为2.15, 2.33, 2.46, 2.56, 2.74