"""
比较不同优化器的性能

本示例比较了Riemann框架中不同优化器在相同优化问题上的性能表现。
使用Rosenbrock函数作为测试函数，该函数是一个经典的非凸优化问题。

主要功能：
1. 实现多种优化器（SGD、Adam、Adagrad、LBFGS等）
2. 在Rosenbrock函数上测试各优化器的收敛速度
3. 可视化优化路径和收敛曲线
4. 比较不同优化器的优缺点
5. 提供优化器选择的参考建议
"""

import os
import sys
import time
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

import riemann as rm
from riemann import optim

# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    np.random.seed(seed)

# 清屏函数
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# 定义Rosenbrock函数（用于二维情况）
def rosenbrock_2d(x, y):
    """Rosenbrock函数（香蕉函数），极小值在(1, 1)处，值为0"""
    # 确保返回标量
    return 100. * (y - x**2.)**2. + (1. - x)**2.

# 使用GD优化器测试
def test_gd():
    print("\n=== 梯度下降法(GD) ===")
    # 初始化参数，从远离最优解的点开始
    x = rm.tensor(-1.2, requires_grad=True)
    y = rm.tensor(1.0, requires_grad=True)
    params = [x, y]
    
    # 优化后的参数配置
    optimizer = optim.GD(params, lr=0.001)
    
    start_time = time.time()
    
    # 优化过程
    max_iterations = 200000
    tolerance = 1e-8
    prev_loss = float('inf')
    
    for i in range(max_iterations):
        # 前向传播计算损失
        loss = rosenbrock_2d(x, y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 检查收敛
        if i % 20000 == 0:
            print(f"Iteration {i}: loss = {loss.item()}, x = {x.item()}, y = {y.item()}")
            
        if abs(prev_loss - loss.item()) < tolerance:
            print(f"Converged at iteration {i}")
            break
        
        prev_loss = loss.item()
    
    end_time = time.time()
    
    print(f"Final loss: {loss.item()}")
    print(f"Final parameters: x = {x.item()}, y = {y.item()}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    
    return loss.item(), (x.item(), y.item()), end_time - start_time

# 使用SGD优化器测试
def test_sgd():
    print("\n=== 随机梯度下降法(SGD) ===")
    # 初始化参数，从远离最优解的点开始
    x = rm.tensor(-1.2, requires_grad=True)
    y = rm.tensor(1.0, requires_grad=True)
    params = [x, y]
    
    # 优化后的参数配置
    optimizer = optim.SGD(params, lr=0.0005, momentum=0.9)
    
    start_time = time.time()
    
    # 优化过程
    max_iterations = 200000
    tolerance = 1e-8
    prev_loss = float('inf')
    
    for i in range(max_iterations):
        # 前向传播计算损失
        loss = rosenbrock_2d(x, y)
        
        # 检查数值稳定性
        current_loss = loss.item()
        if not np.isfinite(current_loss):
            print(f"NaN detected at iteration {i}, stopping optimization")
            break
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 检查参数范围
        if abs(x.item()) > 10 or abs(y.item()) > 10:
            print(f"Parameters out of range at iteration {i}, stopping optimization")
            break
        
        # 检查收敛
        if i % 20000 == 0:
            print(f"Iteration {i}: loss = {current_loss}, x = {x.item()}, y = {y.item()}")
            
        if abs(prev_loss - current_loss) < tolerance:
            print(f"Converged at iteration {i}")
            break
        
        prev_loss = current_loss
    
    end_time = time.time()
    
    print(f"Final loss: {current_loss}")
    print(f"Final parameters: x = {x.item()}, y = {y.item()}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    
    return current_loss, (x.item(), y.item()), end_time - start_time

# 使用Adam优化器测试
def test_adam():
    print("\n=== Adam优化器 ===")
    # 初始化参数，从远离最优解的点开始
    x = rm.tensor(-1.2, requires_grad=True)
    y = rm.tensor(1.0, requires_grad=True)
    params = [x, y]
    
    # 优化后的参数配置
    optimizer = optim.Adam(params, lr=0.05, betas=(0.9, 0.999), eps=1e-8)
    
    start_time = time.time()
    
    # 优化过程
    max_iterations = 20000
    tolerance = 1e-8
    prev_loss = float('inf')
    
    for i in range(max_iterations):
        # 前向传播计算损失
        loss = rosenbrock_2d(x, y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 检查收敛
        if i % 2000 == 0:
            print(f"Iteration {i}: loss = {loss.item()}, x = {x.item()}, y = {y.item()}")
            
        if abs(prev_loss - loss.item()) < tolerance:
            print(f"Converged at iteration {i}")
            break
        
        prev_loss = loss.item()
    
    end_time = time.time()
    
    print(f"Final loss: {loss.item()}")
    print(f"Final parameters: x = {x.item()}, y = {y.item()}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    
    return loss.item(), (x.item(), y.item()), end_time - start_time

# 使用Adagrad优化器测试
def test_adagrad():
    print("\n=== Adagrad优化器 ===")
    # 初始化参数，从远离最优解的点开始
    x = rm.tensor(-1.2, requires_grad=True)
    y = rm.tensor(1.0, requires_grad=True)
    params = [x, y]
    
    # 优化后的参数配置
    optimizer = optim.Adagrad(params, lr=0.3, initial_accumulator_value=0.1, eps=1e-7)
    
    start_time = time.time()
    
    # 优化过程
    max_iterations = 80000
    tolerance = 1e-8
    prev_loss = float('inf')
    
    for i in range(max_iterations):
        # 前向传播计算损失
        loss = rosenbrock_2d(x, y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 检查收敛
        if i % 8000 == 0:
            print(f"Iteration {i}: loss = {loss.item()}, x = {x.item()}, y = {y.item()}")
            
        if abs(prev_loss - loss.item()) < tolerance:
            print(f"Converged at iteration {i}")
            break
        
        prev_loss = loss.item()
    
    end_time = time.time()
    
    print(f"Final loss: {loss.item()}")
    print(f"Final parameters: x = {x.item()}, y = {y.item()}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    
    return loss.item(), (x.item(), y.item()), end_time - start_time

# LBFGS优化器测试函数
def test_lbfgs():
    print("\n=== LBFGS优化器 ===")
    # 初始化参数
    x = rm.tensor(-1.2, requires_grad=True)
    y = rm.tensor(1.0, requires_grad=True)
    params = [x, y]
    
    # 优化后的参数配置
    optimizer = optim.LBFGS(params, lr=1.0, max_iter=50, max_eval=50,
                       tolerance_grad=1e-4, tolerance_change=1e-9,
                       history_size=20)
    
    start_time = time.time()
    
    # LBFGS需要闭包函数
    def closure():
        optimizer.zero_grad(True)
        loss = rosenbrock_2d(x, y)
        loss.backward()
        return loss
    
    # 优化后的外部迭代次数
    max_iterations = 5
    loss = None
    best_loss = float('inf')
    best_x = x.item()
    best_y = y.item()
    
    for i in range(max_iterations):
        loss = optimizer.step(closure)
        
        # 获取损失值
        current_loss = loss.item()
        x_val = x.item()
        y_val = y.item()
        
        # 更新最佳值
        if current_loss < best_loss and np.isfinite(current_loss):
            best_loss = current_loss
            best_x = x_val
            best_y = y_val
        
        # 检查是否溢出
        if not np.isfinite(current_loss) or not np.isfinite(x_val) or not np.isfinite(y_val):
            print(f"数值溢出，参数发散，停止优化")
            break
            
        # 使用更合理的收敛阈值
        if current_loss < 1e-8:
            print(f"已收敛")
            break
            
        # 添加参数值过大的检查
        if abs(x_val) > 10 or abs(y_val) > 10:
            print(f"参数值过大，停止优化")
            break
    
    end_time = time.time()
    
    print(f"最终损失值: {best_loss:.10f}")
    print(f"最终参数: x = {best_x:.10f}, y = {best_y:.10f}")
    print(f"计算时间: {end_time - start_time:.4f} seconds")
    
    return best_loss, (best_x, best_y), end_time - start_time

# 主函数，比较所有优化器
def main():
    print("多元函数极小值优化器比较测试")
    print("测试函数: Rosenbrock函数 f(x,y) = 100*(y-x^2)^2 + (1-x)^2")
    print("理论极小值: f(1,1) = 0")
    
    set_seed(42)
    
    results = {}
    
    # 测试各种优化器
    try:
        results['GD'] = test_gd()
    except Exception as e:
        print(f"GD测试失败: {e}")
        
    try:
        results['SGD'] = test_sgd()
    except Exception as e:
        print(f"SGD测试失败: {e}")
        
    try:
        results['Adam'] = test_adam()
    except Exception as e:
        print(f"Adam测试失败: {e}")
        
    try:
        results['Adagrad'] = test_adagrad()
    except Exception as e:
        print(f"Adagrad测试失败: {e}")
        
    # try:
    results['LBFGS'] = test_lbfgs()
    # except Exception as e:
    #     print(f"LBFGS测试失败: {e}")
    
    # 打印汇总结果
    print("\n" + "="*80)
    print("优化器性能比较汇总")
    print("="*80)
    print("{:<10}|{:<18}|{:<25}|{:<14}".format("优化器", "最终损失值", "参数值", "耗时(秒)"))
    print("-"*80)
    
    for optimizer_name, (loss, params, time_taken) in results.items():
        # 处理可能的NaN值
        if isinstance(loss, float) and not np.isfinite(loss):
            loss_str = "nan"
            params_str = "(nan, nan)"
        else:
            loss_str = "{:.10f}".format(loss)[:18]
            params_str = "({:.8f}, {:.8f}) ".format(params[0], params[1])[:25]
        time_str = "{:.4f}".format(time_taken)
        
        print("{:<13}|{:<23}|{:<28}|{:<14}".format(
            optimizer_name, loss_str, params_str, time_str
        ))
    
    print("="*80)
    
if __name__ == "__main__":
    clear_screen()
    main()