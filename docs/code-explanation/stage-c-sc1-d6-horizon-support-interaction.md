# SC1-D6 Diagnostic Code Explanation

forecast model与head computation沿用D5。唯一数据路径变化是`collect_rows(..., start_batch=8)`：validation loader
先跳过0-7批，再收集8批；train fit/holdout完全不变。metadata新增`validation_batch_offset`，analyzer要求15条均为8。

D6 analyzer固定读取`block_dct2_b144`，分别对short与long horizon windows平均log effects，并统计每个
dataset/checkpoint是否同时出现short-positive与long-negative。b48/b96只保留为描述性support ordering controls，
不参与candidate选择。

Code-theory consistency：代码直接检验“相同full forecast readout在不同prefix domain上的local/global support
偏好是否反转”。它仍是frozen-memory proxy，不证明新operator可实现该折中，也不把H输入learned computation。
