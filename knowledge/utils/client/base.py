"""客户端管理基类：懒加载单例 + 环境变量读取（含 Windows 注册表回退）"""
import logging
import os
import threading

logger = logging.getLogger(__name__)


def _read_registry_env(name: str):
    """从 Windows 系统/用户环境变量（注册表）读取。

    背景：Windows 环境变量在进程启动时快照。若 IDE/终端在配置系统环境变量
    （setx/系统设置）之前已启动，其子进程读不到新变量。此处直接读注册表，
    保证旧进程也能拿到系统级配置。
    """
    if os.name != "nt":
        return None
    try:
        import winreg
    except Exception:
        return None
    for hive, key_path in [
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
        (winreg.HKEY_CURRENT_USER, r"Environment"),
    ]:
        try:
            with winreg.OpenKey(hive, key_path) as key:
                val, _ = winreg.QueryValueEx(key, name)
                if val:
                    return val
        except OSError:
            continue
    return None


def get_env(name: str, default: str = ""):
    """读取环境变量；进程缺失时回退到注册表中的系统/用户变量"""
    value = os.getenv(name)
    if value is None:
        value = _read_registry_env(name)
    return value if value is not None else default


class BaseClientManager:
    """客户端管理器基类：提供线程安全懒加载单例与环境变量校验"""

    @classmethod
    def _get_or_create(cls, attr_name, lock: threading.Lock, factory):
        """懒加载单例：对象已存在直接返回，否则加锁创建"""
        instance = getattr(cls, attr_name, None)
        if instance is not None:
            return instance
        with lock:
            instance = getattr(cls, attr_name, None)
            if instance is None:
                instance = factory()
                setattr(cls, attr_name, instance)
        return instance

    @classmethod
    def _require_env(cls, key: str) -> str:
        """读取必需环境变量，缺失则抛 EnvironmentError"""
        value = get_env(key)
        if not value:
            raise EnvironmentError(f"缺少环境变量: {key}")
        return value
