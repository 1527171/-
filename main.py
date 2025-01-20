import pandas as pd


def load_data():
    goods = pd.read_excel("库存表.xlsx")
    orders = pd.read_excel("购买表.xlsx")
    trans = pd.read_excel("运输表.xlsx")
    return goods, orders, trans


def get_orders(orders):
    n = 5
    orders_items = orders["物品编号"].head(n).tolist()  # 物品编号
    orders_destination = orders["目的地"].head(n).tolist()  # 目的地
    orders_counts = orders["购买数量"].head(n).tolist()  # 购买数量
    orders_time = orders["时限"].head(n).tolist()  # 时限
    return orders_items, orders_destination, orders_counts, orders_time


def select_repos(orders, goods, trans):
    # 从 orders 表中提取相关字段
    orders_items, orders_destination, orders_counts, orders_time = get_orders(orders)
    # 初始化一个空的 DataFrame 来存储候选仓库
    candidate_repos = pd.DataFrame()

    # 遍历订单数据
    for item, destination, count, time_limit in zip(orders_items, orders_destination, orders_counts, orders_time):
        # 筛选出当前物品编号对应的库存记录
        select_item = goods[goods["物品编号"] == item]

        # 如果库存中没有该物品，跳过
        if select_item.empty:
            print(f"物品 {item} 无库存记录")
            continue

        # 遍历筛选出的库存记录
        for _, stock_row in select_item.iterrows():
            # 获取当前库存记录的库存地
            stock_location = stock_row["库存地"]

            # 筛选出从当前库存地到目的地的运输记录
            trans_record = trans[(trans["始发地"] == stock_location) & (trans["目的地"] == destination)]

            # 如果没有运输记录，跳过
            if trans_record.empty:
                print(f"从 {stock_location} 到 {destination} 无运输记录")
                continue

            # 检查时间条件：订单时限 >= 发货时间 + 运输时间
            time_condition = (
                    time_limit >= stock_row["发货时间"] + trans_record.iloc[0]["所需时间（小时）"]
            )

            # 检查数量条件：库存数量 >= 订单购买数量
            quantity_condition = stock_row["数量"] >= count

            # 如果同时满足时间和数量条件，则将该记录加入候选
            if time_condition and quantity_condition:
                # 将符合条件的记录添加到候选 DataFrame
                candidate_repos = pd.concat(
                    [candidate_repos, stock_row.to_frame().T]
                )

    return candidate_repos


def structure_data(candidate_repos):
    # 获取唯一的仓库列表和物品编号列表
    reposs = candidate_repos["库存地"].unique()
    items = candidate_repos["物品编号"].unique()

    # 创建一个空的 DataFrame，行是物品编号，列是仓库
    adjacency_matrix = pd.DataFrame(0, index=items, columns=reposs)

    # 遍历候选仓库记录，填充邻接矩阵
    for _, row in candidate_repos.iterrows():
        item = row["物品编号"]
        repos = row["库存地"]
        adjacency_matrix.loc[item, repos] = 1

    return adjacency_matrix


def find_minimal_cover(adjacency_matrix):
    repos = adjacency_matrix.columns.tolist()  # 仓库列表
    items = adjacency_matrix.index.tolist()  # 商品列表

    # 计算每个仓库能覆盖的商品
    repo_cover = {repo: set(adjacency_matrix.index[adjacency_matrix[repo] == 1].tolist()) for repo in repos}

    minimal_covers = []  # 用一个列表保存所有最小覆盖的候选集合
    min_length = float('inf')  # 记录当前最小覆盖的长度

    # 存储所有状态，避免重复计算
    visited = set()  # 用来存储已经访问过的状态，避免重复计算

    # 定义回溯函数
    def backtrack(current_select_repos, uncovered_items):
        nonlocal min_length, minimal_covers

        # 如果所有商品都被覆盖
        if not uncovered_items:
            if len(current_select_repos) < min_length:
                min_length = len(current_select_repos)
                minimal_covers = [current_select_repos]
            elif len(current_select_repos) == min_length:
                minimal_covers.append(current_select_repos)
            return

        # 通过将已选择仓库集合和未覆盖商品集合作为状态，避免重复计算
        state = (frozenset(current_select_repos), frozenset(uncovered_items))
        if state in visited:
            return
        visited.add(state)

        # 贪心选择能够覆盖最多未覆盖商品的仓库
        best_cover = 0
        candidate_repos = []

        for repo in repos:
            # 剪枝：如果当前仓库未被选择，并且可以覆盖未覆盖的商品
            if repo not in current_select_repos:
                covered_items = repo_cover[repo] & uncovered_items
                cover_len = len(covered_items)

                if cover_len > best_cover:
                    best_cover = cover_len
                    candidate_repos = [repo]
                elif cover_len == best_cover:
                    candidate_repos.append(repo)

        # 递归处理所有候选仓库
        for repo in candidate_repos:
            new_uncovered_items = uncovered_items - repo_cover[repo]
            backtrack(current_select_repos + [repo], new_uncovered_items)

    # 初始调用回溯函数
    backtrack([], set(items))

    return minimal_covers


def calculate_cost(count, trans_record):
    transport_cost = trans_record.iloc[0]["运输成本（元）"]
    warehouse_cost = trans_record.iloc[0]["目的地仓储成本（元）"]
    return (transport_cost + warehouse_cost) * count


def output_best_routes(orders, trans, adjacency_matrix, minimal_cover):
    # 获取订单物品
    orders_items, orders_destination, orders_counts, _ = get_orders(orders)
    total_cost = float('inf')  # 初始化总成本为无穷大
    best_routes = []  # 存储最佳仓库组合
    item_details = {}  # 存储每个商品的详细信息
    unsatisfied_items = set()
    # 去重 minimal_cover 中的组合（内容相同，顺序不同的组合去除）
    unique_cover = []
    seen_combinations = set()  # 用于存储已处理的仓库组合（去重）

    for repos_set in minimal_cover:
        sorted_repos_set = tuple(sorted(repos_set))  # 排序并转换为元组（元组是可哈希的）
        if sorted_repos_set not in seen_combinations:
            unique_cover.append(repos_set)
            seen_combinations.add(sorted_repos_set)

    # 遍历每一个最小覆盖的仓库组合
    for repos_set in unique_cover:
        current_cost = 0  # 当前仓库组合的总成本
        item_to_repo = {}  # 用于存储每个商品的最优仓库
        temp_item_details = {}  # 用于存储当前仓库组合下的商品详细信息

        # 遍历每个商品
        for item, count, destination in zip(orders_items, orders_counts, orders_destination):
            if item not in adjacency_matrix.index:  # 检查商品是否存在于 adjacency_matrix 中
                unsatisfied_items.add(item)
                continue  # 跳过该商品
            min_item_cost = float('inf')  # 当前商品的最小成本
            best_repo = None  # 当前商品的最佳仓库
            best_transport_record = None  # 当前商品的最佳运输记录

            # 找到该商品能由哪些仓库提供
            for repo in repos_set:
                if adjacency_matrix.loc[item, repo] == 1:  # 该仓库能提供该商品
                    # 查找该仓库到目的地的运输记录
                    trans_record = trans[(trans["始发地"] == repo) & (trans["目的地"] == destination)]

                    if not trans_record.empty:
                        # 计算运输和仓储成本
                        total_item_cost = calculate_cost(count, trans_record)

                        # 如果当前成本更小，更新最小成本和最佳仓库
                        if total_item_cost < min_item_cost:
                            min_item_cost = total_item_cost
                            best_repo = repo
                            best_transport_record = trans_record.iloc[0]  # 记录运输途径和相关成本

            # 如果找不到合适的仓库，跳过该商品
            if best_repo is None:
                print(f"未找到适合的仓库来满足物品 {item} 的需求")
                continue

            # 将最小成本加入当前组合的总成本
            current_cost += min_item_cost
            item_to_repo[item] = best_repo  # 记录每个商品对应的仓库

            # 将商品的详细信息保存到临时字典中
            temp_item_details[item] = {
                "始发地": best_repo,
                "目的地": destination,
                "成本": round(min_item_cost, 3),
                "运输途径": best_transport_record["运输途径"]
            }

        # 如果当前组合的成本比当前最小成本还小，则更新最小成本和最佳仓库组合
        if current_cost < total_cost:
            total_cost = current_cost
            best_routes = repos_set  # 更新最小成本对应的仓库组合
            # 更新最终的 item_details，确保它是基于最优仓库组合的
            item_details = temp_item_details
    # 输出没有满足条件的商品
    if unsatisfied_items:
        print(f"以下商品没有满足条件的路径: {list(unsatisfied_items)}")
    return best_routes, total_cost, item_details


def main():
    # 加载数据
    goods, orders, trans = load_data()
    # 调用函数筛选候选仓库
    candidate_repos = select_repos(orders, goods, trans)
    print("候选仓库记录：")
    print(candidate_repos)
    # 调用函数生成邻接矩阵
    adjacency_matrix = structure_data(candidate_repos)
    print("邻接矩阵：")
    print(adjacency_matrix.to_string())
    # 调用函数生成最小覆盖的候选集合
    minimal_cover = find_minimal_cover(adjacency_matrix)
    if len(minimal_cover) <= 100:
        print("最小覆盖的候选集合：")
        print(minimal_cover)
    else:
        print("最小覆盖的候选集合长度:", len(minimal_cover))

    # 输出最佳路径及成本
    best_routes, total_cost, item_details = output_best_routes(orders, trans, adjacency_matrix, minimal_cover)
    print("最优仓库组合：", best_routes)
    print("最小总成本：", round(total_cost, 3))

    # 输出每个商品的详细信息
    print("\n每个商品的详细信息：")
    for item, details in item_details.items():
        print(f"{item}: {details['始发地']} -> {details['目的地']} 成本 {details['成本']} 运输途径 {details['运输途径']}")


if __name__ == "__main__":
    main()
