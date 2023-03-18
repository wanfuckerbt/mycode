import math
class multi_factor_simulator_v5:
    """
    并表maker回测
    """

    def __init__(self, signals_matrix, total_trades_p, total_trades_v, ts_lis, symbol_lis, step_size=1, cash=100000, position=0,
                 order_size=1000, taker_fee=.0003, maker_fee=0, max_leverage=100, min_slippage=0, max_slippage=0,
                 open_ratio=0.2, max_open_time=5*60*1000, latency=15, adjust_mode='fix', 
                 place_freq=5*1000, price_step=1, adjust_step=1, lookback_period=5*60*1000, plot=True):
        """
            时间是ms为单位
        """
        
        # 裁剪信号
        if isinstance(ts_lis[0], (int, np.int64, float)):
            pass
        else:
            ts_lis = (pd.Series(ts_lis).apply(datetime.timestamp).values * 1000).astype(int)
        cond = (ts_lis > total_trades_p.index[0]) & (ts_lis <= total_trades_p.index[-1])
        signals_matrix = signals_matrix[cond, :]
        ts_lis = ts_lis[cond]
        
        # 创建信号容器
        self.trade_signals = {}
        for i in range(len(ts_lis)):
            self.trade_signals[ts_lis[i]] = signals_matrix[i]
            
        self.total_trades_p = total_trades_p
        self.total_trades_v = total_trades_v
        
        self.ts_lis = ts_lis
        self.symbol_lis = symbol_lis
        
        self.lookback_period = lookback_period
        self.adjust_step = adjust_step
        self.step_size = step_size
        self.cash = cash
        
        self.position = position
        self.order_size = order_size
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.max_leverage = max_leverage
        self.max_slippage = max_slippage  # 控制挂高挂低的幅度
        self.min_slippage = min_slippage  # 控制挂高挂低的幅度
        self.open_ratio = open_ratio
        self.max_open_time = max_open_time

        self.adjust_mode = adjust_mode
        self.place_freq = place_freq
        self.price_step = price_step
        self.latency = latency
        self.plot = plot
        
    
    def start_loop(self):
        n, m = len(self.ts_lis), len(self.symbol_lis)
        res = np.zeros((n, m + 1))
        res.fill(np.nan)
        filled_amount_matrix = np.zeros((n, m))
        filled_volume_matrix = np.zeros((n, m))
        fee_matrix = np.zeros((n, m))
        open_time_matrix = np.zeros((n, m))
        success_weight_ratio_matrix = np.zeros((n, m))
        filled_avg_price_matrix = np.zeros((n, m))
        mark_price_matrix = np.zeros((n, m))
        success_weight_ratio_stats_lis = []
        present_price = [0] * m
        
        trade_weight_matrix = np.zeros((n, m))

        # 对齐信号的symbol_lis和trades的symbol_lis
        self.total_trades_p = total_trades_p.loc[:, self.symbol_lis].astype(float)
        self.total_trades_v = total_trades_v.loc[:, self.symbol_lis].astype(float)
        _cnt = 0
        
        for i in range(n):
            
            ts = self.ts_lis[i]
            signal = self.trade_signals[ts]
            
            # 截取该信号开仓时间内的trades
            trades_p_flow = self.total_trades_p[(self.total_trades_p.index>=ts) & (self.total_trades_p.index<=ts+self.max_open_time)]
            trades_v_flow = self.total_trades_v[(self.total_trades_v.index>=ts) & (self.total_trades_v.index<=ts+self.max_open_time)]
            
            if self.adjust_mode == 'fix':
            
                # 计算目标weight
                if np.nansum(np.abs(signal)) == 0:
                    allocated_weight = np.array([0] * m)
                else:
                    allocated_weight = np.array([signal[i] / np.nansum(np.abs(signal)) for i in range(m)])

                # 计算当前weight
                present_price = trades_p_flow.iloc[0].values
                present_amount = np.abs(self.position) * present_price
                if np.sum(present_amount) == 0:
                    tmp_weight = [0] * m
                else:
                    tmp_weight = self.position * present_price / self.order_size

                # 计算要达到allocated_weight 需要交易的实际仓位
                trade_weight = allocated_weight - tmp_weight

                """
                开始撮合
                """

                # 控制价格精度
                price_precision = np.array([1 / 10 ** len(str(price).split(".")[-1]) if "." in str(price) else 1.0 for price in present_price])

                # 构建挂单价格
                place_price = [present_price[j] - self.price_step * price_precision[j] if trade_weight[j] > 0 else present_price[j] + self.price_step * price_precision[j] if trade_weight[j] < 0 else present_price[j] for j in range(m)]

                # 构建orders
                orders_p = np.array(place_price)
                orders_v = np.array(trade_weight)*self.order_size

                # 构建指标统计容器
                filled_volume = np.array([0.0] * m)
                filled_amount = np.array([0.0] * m)


                # 开始撮合, 这一步循环非常大需要考虑优化数据结构
                triger_ts = np.array([ts] * m)
                place_ts = np.array([ts] * m)

                open_time = np.array([np.nan] * m)

                filled_state = np.array([1] * m)

                for k in tqdm(range(len(trades_p_flow))):

                    trades_p = trades_p_flow.iloc[k].values
                    trades_v = trades_v_flow.iloc[k].values
                    # 更新当前价格
                    present_price = trades_p

                    # 判断当前时间是否需要更新价格
                    present_ts = trades_p_flow.index[k]

                    # 更新时间判断
                    update_ts_cond = (present_ts - triger_ts) >= self.place_freq

                    # 更新挂单价格
                    place_price = np.array([present_price[j] - self.price_step * price_precision[j] if trade_weight[j] > 0 and update_ts_cond[j] else present_price[j] + self.price_step * price_precision[j] if trade_weight[j] < 0 and update_ts_cond[j] else place_price[j] for j in range(m)])
                    self.place_price = place_price
                    orders_p = np.array(place_price)
                    triger_ts[update_ts_cond] = present_ts
                    place_ts = triger_ts

                    # 交易时间判断
                    trade_ts_cond = (present_ts - place_ts - self.latency) > 0
                    self.orders_p, self.trades_p = orders_p, trades_p


                    # long
                    cond = (orders_p >= trades_p) & (orders_v > 0) & (trades_v < 0) & trade_ts_cond & filled_state
                    cond = cond.astype(bool)
                    filled_volume[cond] -= trades_v[cond] * self.open_ratio
                    filled_amount[cond] -= trades_v[cond] * self.open_ratio * place_price[cond]
                    # 更新挂单价格
                    orders_p[cond] = present_price[cond] - self.price_step * price_precision[cond]
                    # 更新挂单时间
                    place_ts[cond] = present_ts
                    triger_ts = place_ts

                    # short
                    cond = (orders_p <= trades_p) & (orders_v < 0) & (trades_v > 0) & trade_ts_cond & filled_state
                    cond = cond.astype(bool)
                    filled_volume[cond] -= trades_v[cond] * self.open_ratio
                    filled_amount[cond] -= trades_v[cond] * self.open_ratio * place_price[cond]
                    # 更新挂单价格
                    orders_p[cond] = present_price[cond] + self.price_step * price_precision[cond]
                    # 更新挂单时间
                    place_ts[cond] = present_ts
                    triger_ts = place_ts

                    # 判断filled_state
                    filled_cond = (np.abs(filled_amount) >= np.abs(orders_v)) & (filled_state == 1) 
                    filled_state[filled_cond] = 0
                    # 除掉多出来的成交
                    filled_volume[filled_cond] += (orders_v[filled_cond] - filled_amount[filled_cond]) / place_price[filled_cond]
                    filled_amount[filled_cond] = orders_v[filled_cond]

                    open_time[filled_cond] = (present_ts - ts) / 1000 / 60
            
            elif self.adjust_mode == 'dynamic_fix':
                # 计算目标weight
                
                # 计算上一时刻的各品种的volume，以此来赋权每个coin的成交率
                if i == 0:
                    # 等权
                    adjust_weight = np.array([1] * m)
                    adjust_weight = adjust_weight / np.nansum(adjust_weight)
                else:
                    # 截取当前时刻信号前的lookback时段的trades的成交量
                    lookback_volume_rank = self.total_trades_v[(self.total_trades_v.index>=ts-self.lookback_period) & (self.total_trades_v.index<ts)].abs().sum(axis=0).rank().values
                    adjust_weight = 1 / lookback_volume_rank
                    adjust_weight = adjust_weight / np.nansum(adjust_weight)
                
                
                if np.nansum(np.abs(signal)) == 0:
                    allocated_weight = np.array([0] * m)
                else:
                    allocated_weight = np.array([signal[i] / np.nansum(np.abs(signal)) for i in range(m)])
                    
                # 计算当前weight
                present_price = trades_p_flow.iloc[0].values
                present_amount = np.abs(self.position) * present_price
                if np.sum(present_amount) == 0:
                    tmp_weight = [0] * m
                else:
                    tmp_weight = self.position * present_price / self.order_size

                # 计算要达到allocated_weight 需要交易的实际仓位
                trade_weight = allocated_weight - tmp_weight
                # 调整allocated_weight
                adjusted_trade_weight = trade_weight * (1 + self.adjust_step * adjust_weight)
                
                
                """
                开始撮合
                """

                # 控制价格精度
                price_precision = np.array([1 / 10 ** len(str(price).split(".")[-1]) if "." in str(price) else 1.0 for price in present_price])

                # 构建挂单价格
                place_price = [present_price[j] - self.price_step * price_precision[j] if adjusted_trade_weight[j] > 0 else present_price[j] + self.price_step * price_precision[j] if adjusted_trade_weight[j] < 0 else present_price[j] for j in range(m)]

                # 构建orders
                orders_p = np.array(place_price)
                orders_v = np.array(adjusted_trade_weight)*self.order_size

                # 构建指标统计容器
                filled_volume = np.array([0.0] * m)
                filled_amount = np.array([0.0] * m)


                # 开始撮合, 这一步循环非常大需要考虑优化数据结构
                triger_ts = np.array([ts] * m)
                place_ts = np.array([ts] * m)

                open_time = np.array([np.nan] * m)

                filled_state = np.array([1] * m)

                for k in tqdm(range(len(trades_p_flow))):

                    trades_p = trades_p_flow.iloc[k].values
                    trades_v = trades_v_flow.iloc[k].values
                    # 更新当前价格
                    present_price = trades_p

                    # 判断当前时间是否需要更新价格
                    present_ts = trades_p_flow.index[k]

                    # 更新时间判断
                    update_ts_cond = (present_ts - triger_ts) >= self.place_freq

                    # 更新挂单价格
                    place_price = np.array([present_price[j] - self.price_step * price_precision[j] if adjusted_trade_weight[j] > 0 and update_ts_cond[j] else present_price[j] + self.price_step * price_precision[j] if adjusted_trade_weight[j] < 0 and update_ts_cond[j] else place_price[j] for j in range(m)])
                    self.place_price = place_price
                    orders_p = np.array(place_price)
                    triger_ts[update_ts_cond] = present_ts
                    place_ts = triger_ts

                    # 交易时间判断
                    trade_ts_cond = (present_ts - place_ts - self.latency) > 0
                    self.orders_p, self.trades_p = orders_p, trades_p


                    # long
                    cond = (orders_p >= trades_p) & (orders_v > 0) & (trades_v < 0) & trade_ts_cond & filled_state
                    cond = cond.astype(bool)
                    filled_volume[cond] -= trades_v[cond] * self.open_ratio
                    filled_amount[cond] -= trades_v[cond] * self.open_ratio * place_price[cond]
                    # 更新挂单价格
                    orders_p[cond] = present_price[cond] - self.price_step * price_precision[cond]
                    # 更新挂单时间
                    place_ts[cond] = present_ts
                    triger_ts = place_ts

                    # short
                    cond = (orders_p <= trades_p) & (orders_v < 0) & (trades_v > 0) & trade_ts_cond & filled_state
                    cond = cond.astype(bool)
                    filled_volume[cond] -= trades_v[cond] * self.open_ratio
                    filled_amount[cond] -= trades_v[cond] * self.open_ratio * place_price[cond]
                    # 更新挂单价格
                    orders_p[cond] = present_price[cond] + self.price_step * price_precision[cond]
                    # 更新挂单时间
                    place_ts[cond] = present_ts
                    triger_ts = place_ts
                    
                    # 判断filled_state
                    filled_cond = (np.abs(filled_amount) >= np.abs(orders_v)) & (filled_state == 1) 
                    filled_state[filled_cond] = 0
                    # 除掉多出来的成交
                    filled_volume[filled_cond] += (orders_v[filled_cond] - filled_amount[filled_cond]) / place_price[filled_cond]
                    filled_amount[filled_cond] = orders_v[filled_cond]

                    open_time[filled_cond] = (present_ts - ts) / 1000 / 60
                    
                    # 多空仓位判断
                    if np.sum(filled_amount[filled_amount>0]) >= np.sum(trade_weight[trade_weight>0]) * self.order_size:
                        filled_state[filled_amount>0] == 0
                    elif np.sum(np.abs(filled_amount[filled_amount<0])) >= np.sum(np.abs(trade_weight[trade_weight>0])) * self.order_size:
                        filled_state[filled_amount<0] == 0
                    
                    
                    
            
            # 结算指标
            filled_avg_price = filled_amount / filled_volume
            filled_avg_price[trade_weight == 0] = trades_p_flow.iloc[0].values[trade_weight == 0]
            filled_avg_price[np.isnan(filled_avg_price)] = trades_p_flow.iloc[0].values[np.isnan(filled_avg_price)] # filled_volume=0的用初始价格填充
            filled_avg_price[[math.isinf(each) for each in filled_avg_price]] = trades_p_flow.iloc[0].values[[math.isinf(each) for each in filled_avg_price]]
            # print(filled_amount[[math.isinf(each) for each in filled_avg_price]])
            
            success_weight_ratio = abs(filled_amount / (trade_weight * self.order_size))
            success_weight_ratio[np.isnan(success_weight_ratio)] = np.nan # 不操作的coin用np.nan填充
            if i == n-1:
                mark_price = self.total_trades_p[(self.total_trades_p.index>=self.ts_lis[i])].iloc[-1].values
            else:
                mark_price = self.total_trades_p[(self.total_trades_p.index>=self.ts_lis[i]) & (self.total_trades_p.index<=self.ts_lis[i+1])].iloc[-1].values
            open_time[np.isnan(open_time)] = self.max_open_time / 1000 / 60
            open_time[trade_weight == 0] = np.nan
            
            fee_lis = np.array([self.maker_fee] * m)
            
            trade_weight_matrix[i]= trade_weight
            mark_price_matrix[i]=mark_price
            fee_matrix[i] = fee_lis
            filled_avg_price_matrix[i] = filled_avg_price
            open_time_matrix[i] = open_time
            success_weight_ratio_matrix[i] = success_weight_ratio
            success_weight_ratio_stats_lis.append(np.abs(filled_amount).sum() / (np.abs(trade_weight).sum() * self.order_size))
            print(f"需要交易的比重{np.sum(np.abs(trade_weight))} 成交率{np.abs(filled_amount).sum() / (np.abs(trade_weight).sum() * self.order_size)}")
            
            
            """
            风险监控 止盈止损
            """
            
            
            
            
            _cnt += 1
            if _cnt == self.step_size:
                # long: cash减少order_size元，position增加order_size * (1 - fee) / price的coin
                # short: cash增加100元，position减少order_size * (1 + fee) / price的coin
                # trade是每个coin此次调仓的实际收支金额
                success_weight_ratio = np.array([0 if np.isnan(ratio) else ratio for ratio in success_weight_ratio])
                
                
                #现金变化
                trade = -filled_amount * np.where(trade_weight>0,1+fee_matrix[i],1-fee_matrix[i])

                # 实际的仓位增减
                position_change = filled_amount * np.where(filled_avg_price_matrix[i]==0,0,1/filled_avg_price_matrix[i])

                self.filled_amount = filled_amount
                self.trade_weight = trade_weight
                self.success_weight_ratio = success_weight_ratio
                #现金变化
                # trade = -self.order_size * trade_weight * success_weight_ratio * np.where(trade_weight>0,1-fee_matrix[i],1+fee_matrix[i])

                # 实际的仓位增减
                # position_change = self.order_size * trade_weight * success_weight_ratio * np.where(filled_avg_price_matrix[i]==0,0,1/filled_avg_price_matrix[i])

                
                # print(np.sum(self.trade_1 - self.trade_2), np.sum(self.position_change_1 - self.position_change_2))
                tmp_position = self.position + position_change
                tmp_cash = self.cash + np.nansum(trade)

                tmp_equity = np.nansum(tmp_position * mark_price_matrix[i]) + tmp_cash

                leverage = np.nansum(abs(tmp_position) * filled_avg_price_matrix[i])/tmp_equity

                if leverage < self.max_leverage:
                    self.position = tmp_position
                    self.cash = tmp_cash
                    
                    res[i] = np.append(self.position, self.cash)
                else:
                    res[i] = np.append(self.position, self.cash)

                _cnt = 0

        self.res = res
        self.fee_matrix = fee_matrix
        self.filled_avg_price_matrix = filled_avg_price_matrix
        self.trade_weight_matrix = trade_weight_matrix
        self.open_time_matrix = open_time_matrix
        self.success_weight_ratio_matrix = success_weight_ratio_matrix
        self.mark_price_matrix=mark_price_matrix
        self.success_weight_ratio_stats_lis = success_weight_ratio_stats_lis
        self.filled_amount_matrix = filled_amount_matrix
        
        # return res, fee_matrix, filled_avg_price_matrix, open_time_matrix, success_weight_ratio_matrix
    
    def result_analysis(self):
        
        self.position_matrix = self.res[:, :-1].copy()
        self.res[:, :-1] = self.res[:, :-1] * self.mark_price_matrix  # 下一时刻开盘价
        if not isinstance(self.symbol_lis, list):
            self.symbol_lis = self.symbol_lis.tolist()
        self.ts_lis = pd.to_datetime(self.ts_lis, utc=True, unit='ms')
        self.res = pd.DataFrame(self.res, index=self.ts_lis, columns=self.symbol_lis + ['cash'])
        
        
        self.res['equity'] = self.res.sum(axis=1)
        self.res['net_leverage'] = self.res.loc[:, self.symbol_lis].sum(axis=1)/ self.res.equity
        self.res['total_leverage'] = self.res.loc[:, self.symbol_lis].abs().sum(axis=1) / self.res.equity
        # 最后再dropna，否则index可能出现不匹配（）
        self.res.dropna(inplace=True)
        self.position_df = pd.DataFrame(self.position_matrix, index=self.ts_lis, columns=self.symbol_lis)
        
        self.filled_avg_price_df = pd.DataFrame(self.filled_avg_price_matrix, index=self.ts_lis,
                                                    columns=self.symbol_lis).replace([0],[np.nan])
        self.open_time_df = pd.DataFrame(self.open_time_matrix, index=self.ts_lis,
                                                    columns=self.symbol_lis)
        self.success_weight_ratio_df = pd.DataFrame(self.success_weight_ratio_matrix, index=self.ts_lis,
                                                    columns=self.symbol_lis)
        self.fee_df = pd.DataFrame(self.fee_matrix, index=self.ts_lis, columns=self.symbol_lis)
        self.trade_weight_df = pd.DataFrame(self.trade_weight_matrix, index=self.ts_lis, columns=self.symbol_lis)
        self.mark_price_df = pd.DataFrame(self.mark_price_matrix, index=self.ts_lis,
                                                    columns=self.symbol_lis)
        time_span = self.res.index[-1] - self.res.index[0]

        position_usd_change = self.position_df.diff() * self.filled_avg_price_df
        
        # 手续费计算
        maker_fee_cost = np.nansum(position_usd_change.abs()[self.fee_matrix == self.maker_fee]) * self.maker_fee
        taker_fee_cost = np.nansum(position_usd_change.abs()[self.fee_matrix == self.taker_fee]) * self.taker_fee
        # maker_taker_ratio_percentage = np.nansum(fee_matrix == self.maker_fee) / np.nansum(fee_matrix == self.taker_fee) * 100

        n_trades = (self.position_df.loc[:, self.symbol_lis].diff() != 0).sum().sum()
        profit = self.res.equity.iloc[-1] - self.res.equity.iloc[0]
        pnl_percentage = profit / self.res.equity.iloc[0] * 100
        equity_cummax = self.res.equity.cummax()
        max_drawdown_percentage = - (self.res.equity / equity_cummax - 1).min() * 100
        total_traded_quantity = np.nansum(position_usd_change.abs())
        # equity_return = self.res.equity / self.res.equity.shift(1) - 1
        # annual_sharpe_ratio = np.mean(equity_return) / np.std(equity_return) * np.sqrt(float(pd.Timedelta(days=365) / time_span))
        hourly_equity = self.res.resample('1h').agg({'equity': 'last'})
        hourly_equity.dropna(inplace=True)
        hourly_return = hourly_equity.diff()/hourly_equity.shift(1).abs()
        hourly_return.dropna(inplace=True)
        annual_sharpe_ratio = hourly_return.mean() / hourly_return.std() * np.sqrt(
            float(pd.Timedelta(days=365) / pd.Timedelta(hours=1)))

        turnover = (self.position_df.diff().abs() * self.filled_avg_price_df).resample('1h').agg('sum') / self.order_size
        

        equity_change = (self.res['equity'] - self.res['equity'].shift(1)).dropna()
        win_rate = len(equity_change[equity_change > 0]) / len(equity_change) * 100

        # 平均盈亏比
        avg_pnl_ratio = equity_change[equity_change > 0].mean() / np.abs(equity_change[equity_change < 0].mean()) * 100

        convert_to_per_second = float(pd.Timedelta(seconds=1) / time_span)

        # 平均开仓时间统计
        open_time_mean = self.open_time_df.mean().mean()
        # 成交比率统计
        # success_weight_ratio_mean = self.success_weight_ratio_df.replace([0],[np.nan]).mean().mean() * 100
        success_weight_ratio_mean = np.mean(self.success_weight_ratio_stats_lis) * 100
        
        # 收集重要指标
        self.annual_sharpe_ratio = annual_sharpe_ratio[0]
        self.annual_pnl = pnl_percentage * float(pd.Timedelta(days=365) / time_span)
        self.success_weight_ratio_mean = success_weight_ratio_mean
        
        
        print("open time stats for every coins (minutes)")
        display(self.open_time_df.describe().style.background_gradient(cmap='coolwarm'))

        print("open success retion stats for every coins (%)")
        display(self.success_weight_ratio_df.describe().style.background_gradient(cmap='coolwarm'))


        print(
            (
                '%d 小时, %d 笔交易, %.3f USDT, PnL %.2f USDT = %.4f%%\n' \
                '平均每秒 %.2f 笔交易, 平均每秒 %.2f 次更新, 平均每次换仓 PnL: %.7f%%, 最大多仓: %.3f, 最大空仓: %.3f\n' \
                'maker手续费损耗: %.2f, taker手续费损耗: %.2f, 成交比率: %.4f%%, 平均开仓时间: %f 分钟\n' \
                 '最大总杠杆: %.2f,最小总杠杆: %.2f,最大多头风险头寸杠杆：%.2f, 最大空头风险头寸杠杆：%.2f \n'\
                '最大回撤: %.3f%%,交易胜率: %.3f%%,平均盈亏比: %.3f%% 年化 PnL: %.2f%%\n' \
                '月化交易量: %d USDT, 每小时总换手率: %.4f, 每小时各标的的平均换手率: %.4f, 年化夏普率: %.2f\n'
            )
            %
            (
                time_span / pd.Timedelta(hours=1), n_trades, total_traded_quantity, profit, pnl_percentage,

                n_trades * convert_to_per_second, len(self.res) * convert_to_per_second, pnl_percentage / len(self.res),

                self.res.loc[:, self.symbol_lis].sum(axis=1).max(), self.res.loc[:, self.symbol_lis].sum(axis=1).min(),
                maker_fee_cost, taker_fee_cost, success_weight_ratio_mean, open_time_mean,

                 self.res.total_leverage.max(),self.res.total_leverage.min(),self.res.net_leverage.max(),self.res.net_leverage.min()\
                ,max_drawdown_percentage,win_rate, avg_pnl_ratio,
                pnl_percentage * float(pd.Timedelta(days=365) / time_span),

                total_traded_quantity * float(pd.Timedelta(days=30) / time_span),
                turnover.mean(axis=0).sum(), turnover.mean(axis=0).mean(), annual_sharpe_ratio
            )
        )
        print(f"各标的每小时换手率如下 (单位：1)")
        display(turnover.describe().style.background_gradient(cmap='coolwarm'))

        if self.plot:
            # plot
            fig = plt.figure(figsize=(20, 10))
            ax1 = fig.add_subplot(211)
            (self.filled_avg_price_df / self.filled_avg_price_df.shift(1)).cumprod().fillna(1).iloc[49:].plot(ax=ax1, title='')
            plt.legend()

            ax2 = fig.add_subplot(212)
            col = self.symbol_lis

            position_ma = self.res.loc[:, col].rolling(50, min_periods=50).mean()
            position_ma['equity'] = self.res['equity']
            position_ma.iloc[49:].plot(secondary_y=['equity'], ax=ax2)
            plt.legend()
            plt.show()