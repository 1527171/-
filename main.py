import pandas as pd
import os
import sqlite3


def load_data():
    files = ["库存表.xlsx", "购买表.xlsx", "运输表.xlsx", "物品表.xlsx"]
    for file in files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"文件 {file} 不存在，请检查文件路径。")

    goods = pd.read_excel("库存表.xlsx")
    orders = pd.read_excel("购买表.xlsx")
    trans = pd.read_excel("运输表.xlsx")
    name = pd.read_excel("物品表.xlsx")

    # 检查数据是否为空
    if goods.empty or orders.empty or trans.empty or name.empty:
        raise ValueError("其中一个数据表为空，请检查数据文件内容。")

    return goods, orders, trans, name


def load_data_from_db(db_path):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库文件 {db_path} 不存在，请检查路径。")

    conn = sqlite3.connect(db_path)

    try:
        tables = ["库存表", "购买表", "运输表", "物品表"]
        data = {}

        for table in tables:
            query = f"SELECT * FROM {table}"
            df = pd.read_sql_query(query, conn)
            if df.empty:
                raise ValueError(f"数据库表 {table} 为空，请检查数据库内容。")
            data[table] = df

    except Exception as e:
        raise RuntimeError(f"从数据库加载数据时出错: {e}")

    finally:
        conn.close()

    return data["库存表"], data["购买表"], data["运输表"], data["物品表"]


def get_orders(orders):
    if orders is None or orders.empty:
        raise ValueError("订单数据为空，请检查输入数据。")

    required_columns = ["物品编号", "目的地", "购买数量", "时限"]
    for col in required_columns:
        if col not in orders.columns:
            raise KeyError(f"订单数据缺少必要列: {col}")

    n = min(5, len(orders))  # 确保不会超出数据范围

    orders_items = orders["物品编号"].head(n).tolist()
    orders_destination = orders["目的地"].head(n).tolist()
    orders_counts = orders["购买数量"].head(n).tolist()
    orders_time = orders["时限"].head(n).tolist()

    # 数据有效性检查
    if not all(isinstance(item, (int, str)) for item in orders_items):
        raise ValueError("订单数据中的 '物品编号' 包含无效值，应为字符串或整数。")
    if not all(isinstance(dest, str) for dest in orders_destination):
        raise ValueError("订单数据中的 '目的地' 包含无效值，应为字符串。")
    if not all(isinstance(count, (int, float)) and count > 0 for count in orders_counts):
        raise ValueError("订单数据中的 '购买数量' 包含无效值，应为正数。")
    if not all(isinstance(time, (int, float)) and time > 0 for time in orders_time):
        raise ValueError("订单数据中的 '时限' 包含无效值，应为正数。")

    return orders_items, orders_destination, orders_counts, orders_time


def select_repos(orders, goods, trans):
    if orders is None or orders.empty:
        raise ValueError("订单数据为空，请检查输入数据。")
    if goods is None or goods.empty:
        raise ValueError("库存数据为空，请检查输入数据。")
    if trans is None or trans.empty:
        raise ValueError("运输数据为空，请检查输入数据。")

    required_order_columns = ["物品编号", "目的地", "购买数量", "时限"]
    required_goods_columns = ["物品编号", "库存地", "数量", "发货时间"]
    required_trans_columns = ["始发地", "目的地", "所需时间（小时）", "目的地仓储成本（元）", "运输成本（元）", "运输途径"]

    for col in required_order_columns:
        if col not in orders.columns:
            raise KeyError(f"订单数据缺少必要列: {col}")
    for col in required_goods_columns:
        if col not in goods.columns:
            raise KeyError(f"库存数据缺少必要列: {col}")
    for col in required_trans_columns:
        if col not in trans.columns:
            raise KeyError(f"运输数据缺少必要列: {col}")

    # 从 orders 表中提取相关字段
    orders_items, orders_destination, orders_counts, orders_time = get_orders(orders)
    candidate_repos = pd.DataFrame()  # 初始化候选仓库

    # 遍历每个订单项
    for item, destination, count, time_limit in zip(orders_items, orders_destination, orders_counts, orders_time):
        # 筛选当前物品的库存记录
        select_item = goods[goods["物品编号"] == item]
        if select_item.empty:
            print(f"物品 {item} 无库存记录")
            continue

        candidates = {}  # 用于保存每个始发地的最低运输成本候选

        for _, stock_row in select_item.iterrows():
            stock_loc = stock_row["库存地"]
            stock_qty = stock_row["数量"]
            dispatch_time = stock_row["发货时间"]

            if not isinstance(stock_qty, (int, float)) or stock_qty <= 0:
                print(f"库存地 {stock_loc} 的库存数量无效: {stock_qty}")
                continue
            if not isinstance(dispatch_time, (int, float)) or dispatch_time < 0:
                print(f"库存地 {stock_loc} 的发货时间无效: {dispatch_time}")
                continue

            # 筛选始发地到目的地的运输记录
            trans_filtered = trans[(trans["始发地"] == stock_loc) & (trans["目的地"] == destination)]
            if trans_filtered.empty:
                print(f"从 {stock_loc} 到 {destination} 无运输记录")
                continue

            # 筛选满足时间条件的运输记录
            valid_trans = trans_filtered[trans_filtered["所需时间（小时）"] + dispatch_time <= time_limit]
            if valid_trans.empty:
                print(f"从 {stock_loc} 到 {destination} 无满足时间条件的运输记录")
                continue

            # 找到运输成本最低的记录
            min_cost_trans = valid_trans.loc[valid_trans["运输成本（元）"].idxmin()]

            # 检查库存数量是否足够
            if stock_qty >= count:
                # 构造候选记录（包含运输信息）
                candidate = {
                    "物品编号": item,
                    "始发地": stock_loc,
                    "目的地": destination,
                    "库存": stock_qty,
                    "所需时间（小时）": min_cost_trans["所需时间（小时）"],
                    "目的地仓储成本（元）": min_cost_trans["目的地仓储成本（元）"],
                    "运输成本（元）": min_cost_trans["运输成本（元）"],
                    "运输途径": min_cost_trans["运输途径"]
                }
                candidate_df = pd.DataFrame([candidate])

                # 更新候选字典，保留每个始发地的最低运输成本
                if stock_loc not in candidates or candidates[stock_loc]["运输成本（元）"].iloc[0] > candidate["运输成本（元）"]:
                    candidates[stock_loc] = candidate_df

        # 合并当前订单的候选记录到总结果
        if candidates:
            candidate_repos = pd.concat([candidate_repos, *candidates.values()], ignore_index=True)

    return candidate_repos


def structure_data(candidate_repos):
    """ 生成物品和仓库之间的邻接矩阵 """
    if candidate_repos is None or candidate_repos.empty:
        raise ValueError("候选仓库数据为空，无法构造邻接矩阵。")

    required_columns = ["始发地", "物品编号"]
    for col in required_columns:
        if col not in candidate_repos.columns:
            raise KeyError(f"候选仓库数据缺少必要列: {col}")

    # 获取唯一的仓库列表和物品编号列表
    reposs = candidate_repos["始发地"].unique()
    items = candidate_repos["物品编号"].unique()

    # 创建一个空的 DataFrame，行是物品编号，列是仓库
    adjacency_matrix = pd.DataFrame(0, index=items, columns=reposs)

    # 遍历候选仓库记录，填充邻接矩阵
    for _, row in candidate_repos.iterrows():
        item = row["物品编号"]
        repos = row["始发地"]
        adjacency_matrix.loc[item, repos] = 1

    return adjacency_matrix


def find_minimal_cover(adjacency_matrix):
    """ 采用回溯法寻找最小仓库覆盖集合 """
    if adjacency_matrix is None or adjacency_matrix.empty:
        raise ValueError("邻接矩阵为空，无法计算最小覆盖集。")

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
    """ 计算总运输成本 """
    if not isinstance(count, (int, float)) or count <= 0:
        raise ValueError(f"购买数量 {count} 无效，必须为正数。")

    if trans_record is None or trans_record.empty:
        raise ValueError("运输记录为空，无法计算成本。")

    required_columns = ["运输成本（元）", "目的地仓储成本（元）"]
    for col in required_columns:
        if col not in trans_record.columns:
            raise KeyError(f"运输数据缺少必要列: {col}")

    transport_cost = trans_record.iloc[0]["运输成本（元）"]
    warehouse_cost = trans_record.iloc[0]["目的地仓储成本（元）"]

    if not isinstance(transport_cost, (int, float)) or transport_cost < 0:
        raise ValueError(f"运输成本 {transport_cost} 无效，必须为非负数。")
    if not isinstance(warehouse_cost, (int, float)) or warehouse_cost < 0:
        raise ValueError(f"仓储成本 {warehouse_cost} 无效，必须为非负数。")

    return (transport_cost + warehouse_cost) * count


def output_best_routes(orders, candidate_repos, adjacency_matrix, minimal_cover):
    """ 计算最优的仓库组合及运输路径 """

    # 输入数据检查
    if orders is None or orders.empty:
        raise ValueError("订单表 orders 为空，无法计算最佳路径。")
    if candidate_repos is None or candidate_repos.empty:
        raise ValueError("候选仓库数据 candidate_repos 为空，无法计算最佳路径。")
    if adjacency_matrix is None or adjacency_matrix.empty:
        raise ValueError("邻接矩阵 adjacency_matrix 为空，无法计算最佳路径。")
    if minimal_cover is None or not minimal_cover:
        raise ValueError("最小仓库覆盖集合 minimal_cover 为空，无法计算最佳路径。")

    # 获取订单物品
    orders_items, orders_destination, orders_counts, _ = get_orders(orders)

    # 初始化变量
    total_cost = float('inf')  # 最优总成本
    best_routes = []  # 最优仓库组合
    item_details = {}  # 每个商品的详细信息
    unsatisfied_items = set()  # 不能被满足的商品集合

    # 处理 minimal_cover 去重（去除顺序不同但内容相同的组合）
    unique_cover = []
    seen_combinations = set()

    for repos_set in minimal_cover:
        sorted_repos_set = tuple(sorted(repos_set))  # 统一排序
        if sorted_repos_set not in seen_combinations:
            unique_cover.append(repos_set)
            seen_combinations.add(sorted_repos_set)

    # 遍历每个最小覆盖仓库组合
    for repos_set in unique_cover:
        current_cost = 0  # 当前组合的总成本
        item_to_repo = {}  # 记录每个商品的最佳仓库
        temp_item_details = {}  # 临时存储当前组合的商品详细信息

        # 遍历订单中的商品
        for item, count, destination in zip(orders_items, orders_counts, orders_destination):
            if item not in adjacency_matrix.index:
                unsatisfied_items.add(item)
                continue  # 跳过该商品

            min_item_cost = float('inf')  # 当前商品的最小成本
            best_repo = None  # 记录最佳仓库
            best_transport_record = None  # 记录最佳运输信息

            # 查找能提供该商品的仓库
            for repo in repos_set:
                if adjacency_matrix.loc[item, repo] == 1:
                    # 获取该仓库到目的地的运输记录
                    trans_record = candidate_repos[(candidate_repos["始发地"] == repo) &
                                                   (candidate_repos["目的地"] == destination)]

                    if not trans_record.empty:
                        # 计算运输和仓储成本
                        total_item_cost = calculate_cost(count, trans_record)

                        # 选择成本最低的运输方案
                        if total_item_cost < min_item_cost:
                            min_item_cost = total_item_cost
                            best_repo = repo
                            best_transport_record = trans_record.iloc[0]  # 记录运输途径

            # 如果找不到合适的仓库，记录该商品
            if best_repo is None:
                print(f"未找到适合的仓库来满足物品 {item}（目的地：{destination}）")
                unsatisfied_items.add(item)
                continue

            # 记录该商品的最优仓库和成本
            current_cost += min_item_cost
            item_to_repo[item] = best_repo

            temp_item_details[item] = {
                "始发地": best_repo,
                "目的地": destination,
                "成本": round(min_item_cost, 3),
                "运输途径": best_transport_record["运输途径"]
            }

        # 更新最优解
        if current_cost < total_cost:
            total_cost = current_cost
            best_routes = repos_set
            item_details = temp_item_details

    # 输出无法满足的商品
    if unsatisfied_items:
        print(f"以下商品无法满足需求: {list(unsatisfied_items)}")

    return best_routes, total_cost, item_details


def main():
    """ 物流优化算法主函数 """
    try:
        # 加载数据
        goods, orders, trans, name = load_data()
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 调用函数筛选候选仓库
    candidate_repos = select_repos(orders, goods, trans)
    if candidate_repos.empty:
        print("没有找到合适的候选仓库，终止计算。")
        return

    print("\n候选仓库记录：")
    print(candidate_repos)

    # 调用函数生成邻接矩阵
    adjacency_matrix = structure_data(candidate_repos)
    if adjacency_matrix.empty:
        print("邻接矩阵为空，无法进行最小覆盖计算。")
        return

    print("\n邻接矩阵：")
    print(adjacency_matrix.to_string())

    # 调用函数生成最小覆盖的候选集合
    minimal_cover = find_minimal_cover(adjacency_matrix)
    if minimal_cover:
        if len(minimal_cover) <= 100:
            print("\n最小覆盖的候选集合：")
            print(minimal_cover)
        else:
            print(f"\n最小覆盖的候选集合数量: {len(minimal_cover)}（结果过多，仅输出数量）")
    else:
        print("没有找到最小覆盖集合，终止计算。")
        return

    # 输出最佳路径及成本
    best_routes, total_cost, item_details = output_best_routes(orders, candidate_repos, adjacency_matrix, minimal_cover)

    print("\n最优仓库组合：", best_routes)
    print(f"最小总成本: {round(total_cost, 3)} 元")

    # 创建物品编号到物品名称的映射（转换为小写，避免大小写不匹配）
    item_name_map = {str(item).lower(): name for item, name in zip(name['物品编号'], name['物品名称'])}

    # 输出每个商品的详细信息
    if item_details:
        print("\n每个商品的详细信息：")
        for item, details in item_details.items():
            # 获取物品名称，若找不到则标记为 "未知"
            item_name = item_name_map.get(str(item).lower(), "未知")
            print(
                f" {item} [{item_name}]: {details['始发地']} → {details['目的地']} 成本: {details['成本']} 元 运输途径: {details['运输途径']}")
    else:
        print("未找到符合要求的最佳路径方案。")


if __name__ == "__main__":
    main()
