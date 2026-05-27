"""
PLC硬触发模块 - Modbus TCP通信接口

功能：
1. Modbus TCP客户端，连接PLC控制器
2. 硬触发信号监听（上升沿触发拍照+检测）
3. 触发状态回调（检测完成信号反馈给PLC）
4. 超时保护与断线重连机制
5. 配置项集成到config.yaml

使用pymodbus库实现Modbus TCP通信，支持工业标准PLC通信协议。
"""

import threading
import time
import logging
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum

try:
    from pymodbus.client import ModbusTcpClient
    from pymodbus.exceptions import ModbusException, ConnectionException
    PYMODBUS_AVAILABLE = True
except ImportError:
    PYMODBUS_AVAILABLE = False
    # 创建虚拟类用于类型提示
    class ModbusTcpClient:
        pass
    class ModbusException(Exception):
        pass
    class ConnectionException(Exception):
        pass


class TriggerMode(Enum):
    """触发模式枚举"""
    SOFTWARE = "software"      # 软件触发
    PLC_HARDWARE = "plc_hardware"  # PLC硬触发
    CONTINUOUS = "continuous"  # 连续采集


@dataclass
class PLCConfig:
    """PLC配置数据结构"""
    enabled: bool = False
    host: str = "192.168.1.100"
    port: int = 502
    trigger_address: int = 0      # 触发信号寄存器地址
    feedback_address: int = 1     # 反馈信号寄存器地址
    timeout: float = 5.0          # 通信超时(秒)
    retry_interval: float = 1.0   # 重试间隔(秒)
    max_retries: int = 3          # 最大重试次数
    unit_id: int = 1              # Modbus单元ID
    trigger_edge: str = "rising"  # 触发边沿: rising/falling/both
    debounce_time: float = 0.1    # 防抖时间(秒)


class PLCTrigger:
    """
    PLC硬触发控制器
    
    功能：
    1. 建立与PLC的Modbus TCP连接
    2. 监听触发信号（轮询或事件驱动）
    3. 触发回调机制
    4. 状态反馈
    5. 断线重连
    """
    
    def __init__(self, config: PLCConfig):
        """
        初始化PLC触发控制器
        
        Args:
            config: PLC配置参数
        """
        if not PYMODBUS_AVAILABLE:
            raise ImportError("pymodbus库未安装，请运行: pip install pymodbus")
            
        self.config = config
        self.client: Optional[ModbusTcpClient] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._callbacks: Dict[str, Callable] = {}
        self._last_trigger_state = 0
        self._last_trigger_time = 0.0
        self._connection_attempts = 0
        self._logger = logging.getLogger(__name__)
        
        # 状态变量
        self.connected = False
        self.last_error: Optional[str] = None
        self.stats = {
            "trigger_count": 0,
            "success_count": 0,
            "error_count": 0,
            "reconnect_count": 0,
            "last_trigger_time": None
        }
    
    def connect(self) -> bool:
        """
        连接到PLC
        
        Returns:
            bool: 连接是否成功
        """
        if not self.config.enabled:
            self._logger.info("PLC触发功能已禁用")
            return False
            
        try:
            self._logger.info(f"正在连接到PLC: {self.config.host}:{self.config.port}")
            
            # 创建Modbus TCP客户端
            self.client = ModbusTcpClient(
                host=self.config.host,
                port=self.config.port,
                timeout=self.config.timeout,
                retries=self.config.max_retries,
                retry_on_empty=True,
                unit_id=self.config.unit_id
            )
            
            # 测试连接
            if self.client.connect():
                self.connected = True
                self._connection_attempts = 0
                self.last_error = None
                self._logger.info(f"PLC连接成功: {self.config.host}:{self.config.port}")
                return True
            else:
                self.last_error = "连接失败"
                self._logger.error(f"PLC连接失败: {self.config.host}:{self.config.port}")
                return False
                
        except ConnectionException as e:
            self.last_error = f"网络连接错误: {str(e)}"
            self._logger.error(f"PLC网络连接错误: {e}")
            return False
        except ModbusException as e:
            self.last_error = f"Modbus协议错误: {str(e)}"
            self._logger.error(f"Modbus协议错误: {e}")
            return False
        except Exception as e:
            self.last_error = f"未知错误: {str(e)}"
            self._logger.error(f"PLC连接未知错误: {e}")
            return False
    
    def disconnect(self) -> None:
        """断开PLC连接"""
        if self.client:
            try:
                self.client.close()
                self._logger.info("PLC连接已关闭")
            except Exception as e:
                self._logger.error(f"关闭PLC连接时出错: {e}")
            finally:
                self.client = None
                self.connected = False
    
    def reconnect(self) -> bool:
        """
        断线重连
        
        Returns:
            bool: 重连是否成功
        """
        self._connection_attempts += 1
        self._logger.warning(f"尝试重连PLC (第{self._connection_attempts}次)...")
        
        # 先断开旧连接
        self.disconnect()
        
        # 等待重试间隔
        time.sleep(self.config.retry_interval)
        
        # 尝试重连
        success = self.connect()
        
        if success:
            self.stats["reconnect_count"] += 1
            self._logger.info("PLC重连成功")
        else:
            self._logger.error(f"PLC重连失败 (尝试{self._connection_attempts}次)")
            
            # 超过最大重试次数则停止
            if self._connection_attempts >= self.config.max_retries:
                self._logger.error(f"达到最大重试次数({self.config.max_retries})，停止重连")
                self.stop()
        
        return success
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """
        注册事件回调
        
        Args:
            event: 事件类型，支持: 'trigger', 'connected', 'disconnected', 'error'
            callback: 回调函数
        """
        self._callbacks[event] = callback
        self._logger.debug(f"注册回调: {event}")
    
    def _emit_event(self, event: str, *args, **kwargs) -> None:
        """触发事件回调"""
        if event in self._callbacks:
            try:
                self._callbacks[event](*args, **kwargs)
            except Exception as e:
                self._logger.error(f"回调函数执行错误 ({event}): {e}")
    
    def read_trigger_signal(self) -> Optional[int]:
        """
        读取触发信号
        
        Returns:
            Optional[int]: 触发信号值，None表示读取失败
        """
        if not self.connected or not self.client:
            return None
            
        try:
            # 读取保持寄存器（地址从0开始）
            result = self.client.read_holding_registers(
                address=self.config.trigger_address,
                count=1,
                slave=self.config.unit_id
            )
            
            if result.isError():
                self.last_error = f"读取寄存器错误: {result}"
                self._logger.error(f"读取触发信号错误: {result}")
                return None
                
            # 返回寄存器值
            return result.registers[0] if result.registers else 0
            
        except ConnectionException as e:
            self.last_error = f"连接中断: {str(e)}"
            self._logger.error(f"PLC连接中断: {e}")
            self.connected = False
            return None
        except ModbusException as e:
            self.last_error = f"Modbus错误: {str(e)}"
            self._logger.error(f"Modbus通信错误: {e}")
            return None
        except Exception as e:
            self.last_error = f"未知错误: {str(e)}"
            self._logger.error(f"读取触发信号未知错误: {e}")
            return None
    
    def write_feedback_signal(self, value: int) -> bool:
        """
        写入反馈信号
        
        Args:
            value: 反馈值，通常0=未完成，1=完成
            
        Returns:
            bool: 写入是否成功
        """
        if not self.connected or not self.client:
            return False
            
        try:
            # 写入保持寄存器
            result = self.client.write_register(
                address=self.config.feedback_address,
                value=value,
                slave=self.config.unit_id
            )
            
            if result.isError():
                self.last_error = f"写入寄存器错误: {result}"
                self._logger.error(f"写入反馈信号错误: {result}")
                return False
                
            self._logger.debug(f"反馈信号已写入: 地址={self.config.feedback_address}, 值={value}")
            return True
            
        except ConnectionException as e:
            self.last_error = f"连接中断: {str(e)}"
            self._logger.error(f"PLC连接中断: {e}")
            self.connected = False
            return False
        except ModbusException as e:
            self.last_error = f"Modbus错误: {str(e)}"
            self._logger.error(f"Modbus通信错误: {e}")
            return False
        except Exception as e:
            self.last_error = f"未知错误: {str(e)}"
            self._logger.error(f"写入反馈信号未知错误: {e}")
            return False
    
    def send_detection_complete(self, success: bool = True) -> bool:
        """
        发送检测完成信号
        
        Args:
            success: 检测是否成功
            
        Returns:
            bool: 发送是否成功
        """
        feedback_value = 1 if success else 2  # 1=成功完成，2=检测失败
        return self.write_feedback_signal(feedback_value)
    
    def _check_trigger_edge(self, current_value: int) -> bool:
        """
        检查触发边沿
        
        Args:
            current_value: 当前触发信号值
            
        Returns:
            bool: 是否检测到有效触发
        """
        if current_value is None:
            return False
            
        current_time = time.time()
        
        # 防抖处理
        if current_time - self._last_trigger_time < self.config.debounce_time:
            return False
        
        # 检查边沿触发
        if self.config.trigger_edge == "rising":
            # 上升沿触发：0 -> 1
            if self._last_trigger_state == 0 and current_value == 1:
                self._last_trigger_state = current_value
                self._last_trigger_time = current_time
                return True
                
        elif self.config.trigger_edge == "falling":
            # 下降沿触发：1 -> 0
            if self._last_trigger_state == 1 and current_value == 0:
                self._last_trigger_state = current_value
                self._last_trigger_time = current_time
                return True
                
        elif self.config.trigger_edge == "both":
            # 双边沿触发：值发生变化
            if self._last_trigger_state != current_value:
                self._last_trigger_state = current_value
                self._last_trigger_time = current_time
                return True
        
        # 更新最后状态
        self._last_trigger_state = current_value
        return False
    
    def _monitor_loop(self) -> None:
        """PLC监控循环"""
        self._logger.info("PLC监控线程启动")
        
        poll_interval = 0.01  # 10ms轮询间隔，可根据需要调整
        
        while self._running:
            try:
                # 检查连接状态
                if not self.connected:
                    if not self.reconnect():
                        time.sleep(self.config.retry_interval)
                        continue
                
                # 读取触发信号
                trigger_value = self.read_trigger_signal()
                
                if trigger_value is not None:
                    # 检查触发边沿
                    if self._check_trigger_edge(trigger_value):
                        self.stats["trigger_count"] += 1
                        self.stats["last_trigger_time"] = time.time()
                        self._logger.info(f"检测到PLC触发信号 (值={trigger_value})")
                        
                        # 触发回调
                        self._emit_event('trigger', trigger_value)
                
                # 短暂休眠，避免CPU占用过高
                time.sleep(poll_interval)
                
            except Exception as e:
                self._logger.error(f"PLC监控循环错误: {e}")
                self.last_error = str(e)
                time.sleep(1.0)  # 出错后等待1秒
    
    def start(self) -> bool:
        """
        启动PLC监控
        
        Returns:
            bool: 启动是否成功
        """
        if not self.config.enabled:
            self._logger.info("PLC触发功能已禁用，跳过启动")
            return False
            
        if self._running:
            self._logger.warning("PLC监控已在运行")
            return True
            
        # 建立连接
        if not self.connect():
            self._logger.error("PLC连接失败，无法启动监控")
            return False
        
        # 启动监控线程
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        
        self._logger.info("PLC监控已启动")
        self._emit_event('connected')
        return True
    
    def stop(self) -> None:
        """停止PLC监控"""
        if not self._running:
            return
            
        self._running = False
        
        # 等待线程结束
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None
        
        # 断开连接
        self.disconnect()
        
        self._logger.info("PLC监控已停止")
        self._emit_event('disconnected')
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取PLC状态
        
        Returns:
            Dict[str, Any]: 状态信息
        """
        return {
            "enabled": self.config.enabled,
            "connected": self.connected,
            "host": self.config.host,
            "port": self.config.port,
            "last_error": self.last_error,
            "connection_attempts": self._connection_attempts,
            "stats": self.stats.copy(),
            "trigger_address": self.config.trigger_address,
            "feedback_address": self.config.feedback_address,
            "trigger_edge": self.config.trigger_edge
        }
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# 工厂函数，便于从配置创建
def create_plc_trigger_from_config(config_dict: Dict[str, Any]) -> Optional[PLCTrigger]:
    """
    从配置字典创建PLC触发器
    
    Args:
        config_dict: 配置字典，通常来自config.yaml
        
    Returns:
        Optional[PLCTrigger]: PLC触发器实例，如果配置禁用则返回None
    """
    if not config_dict.get("enabled", False):
        return None
        
    try:
        # 从配置字典创建PLCConfig
        plc_config = PLCConfig(
            enabled=config_dict.get("enabled", False),
            host=config_dict.get("host", "192.168.1.100"),
            port=config_dict.get("port", 502),
            trigger_address=config_dict.get("trigger_address", 0),
            feedback_address=config_dict.get("feedback_address", 1),
            timeout=config_dict.get("timeout", 5.0),
            retry_interval=config_dict.get("retry_interval", 1.0),
            max_retries=config_dict.get("max_retries", 3),
            unit_id=config_dict.get("unit_id", 1),
            trigger_edge=config_dict.get("trigger_edge", "rising"),
            debounce_time=config_dict.get("debounce_time", 0.1)
        )
        
        return PLCTrigger(plc_config)
        
    except Exception as e:
        logging.getLogger(__name__).error(f"创建PLC触发器失败: {e}")
        return None


# 测试代码
if __name__ == "__main__":
    import sys
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 测试配置
    test_config = PLCConfig(
        enabled=True,
        host="127.0.0.1",  # 本地测试
        port=502,
        trigger_address=0,
        feedback_address=1,
        timeout=2.0,
        retry_interval=1.0,
        max_retries=3
    )
    
    print("=== PLC硬触发模块测试 ===")
    print(f"配置: {test_config}")
    
    try:
        # 创建触发器
        trigger = PLCTrigger(test_config)
        
        # 注册回调
        def on_trigger(value):
            print(f"[回调] 触发信号: {value}")
            # 模拟检测完成后发送反馈
            success = trigger.send_detection_complete(True)
            print(f"[回调] 反馈信号发送: {'成功' if success else '失败'}")
        
        def on_connected():
            print("[回调] PLC已连接")
        
        def on_disconnected():
            print("[回调] PLC已断开")
        
        trigger.register_callback('trigger', on_trigger)
        trigger.register_callback('connected', on_connected)
        trigger.register_callback('disconnected', on_disconnected)
        
        # 启动
        if trigger.start():
            print("PLC监控已启动，等待触发信号...")
            print("按Ctrl+C停止")
            
            try:
                # 主循环
                while True:
                    time.sleep(1.0)
                    # 定期打印状态
                    status = trigger.get_status()
                    print(f"状态: 连接={status['connected']}, 触发次数={status['stats']['trigger_count']}")
                    
            except KeyboardInterrupt:
                print("\n收到停止信号")
            finally:
                trigger.stop()
        else:
            print("PLC监控启动失败")
            
    except ImportError as e:
        print(f"依赖错误: {e}")
        print("请安装pymodbus: pip install pymodbus")
        sys.exit(1)
    except Exception as e:
        print(f"测试错误: {e}")
        sys.exit(1)