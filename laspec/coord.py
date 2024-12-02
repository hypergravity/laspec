import re


def hex2deg(
    hex: str = "19 28 14.0963893920 +43 55 30.933645492",
) -> tuple[float, float]:
    """Convert hexadecimal to decimal."""
    # 定义用于匹配赤经赤纬各部分的正则表达式模式
    pattern = re.compile(r"(\d+)\s+(\d+)\s+([\d.]+)\s+([+-]?\d+)\s+(\d+)\s+([\d.]+)")

    # 使用正则表达式进行匹配
    match = pattern.match(hex)
    if match is None:
        raise ValueError("Invalid hexadecimal format")

    # 提取匹配到的各个部分
    ra_hours = int(match.group(1))
    ra_minutes = int(match.group(2))
    ra_seconds = float(match.group(3))
    dec_degrees = int(match.group(4))
    dec_minutes = int(match.group(5))
    dec_seconds = float(match.group(6))

    # calculate ra and dec in deg
    ra_deg = (ra_hours + ra_minutes / 60 + ra_seconds / 3600) * 15
    dec_deg = (
        dec_degrees + dec_minutes / 60 + dec_seconds / 3600
        if dec_degrees >= 0
        else -(abs(dec_degrees) + dec_minutes / 60 + dec_seconds / 3600)
    )
    return ra_deg, dec_deg


def jhex2deg(
    jhex: str = "J064726.41+223431.7",
) -> tuple[float, float]:
    """Convert J hexadecimal to decimal."""

    if not jhex.startswith("J"):
        raise ValueError("Invalid hexadecimal format")

    idx_dec_sign = max(jhex.find("+"), jhex.find("-"))
    if idx_dec_sign == -1:
        raise ValueError("Invalid hexadecimal format")

    # 提取匹配到的各个部分
    ra_hours = int(jhex[1:3])
    ra_minutes = int(jhex[3:5])
    ra_seconds = float(jhex[5:idx_dec_sign])
    dec_degrees = int(jhex[idx_dec_sign : idx_dec_sign + 3])
    dec_minutes = int(jhex[idx_dec_sign + 3 : idx_dec_sign + 5])
    dec_seconds = float(jhex[idx_dec_sign + 5 :])

    # calculate ra and dec in deg
    ra_deg = (ra_hours + ra_minutes / 60 + ra_seconds / 3600) * 15
    dec_deg = (
        dec_degrees + dec_minutes / 60 + dec_seconds / 3600
        if dec_degrees >= 0
        else -(abs(dec_degrees) + dec_minutes / 60 + dec_seconds / 3600)
    )
    return ra_deg, dec_deg
