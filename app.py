# -*- coding: utf-8 -*-
"""
马克 33 号：图片参数体检专家 — Streamlit 应用
"""

from __future__ import annotations

import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image


def _parse_named_inches_from_stem(stem: str) -> tuple[int, int] | None:
    """
    从文件名主体（不含后缀）解析宽高（英寸）。取最后一个 '-' 之后的 3/4 位数字。
    """
    if not stem or "-" not in stem:
        return None
    tail = stem.rsplit("-", 1)[-1].strip()
    if not re.fullmatch(r"\d{3}|\d{4}", tail):
        return None
    if len(tail) == 4:
        return int(tail[:2]), int(tail[2:4])
    return int(tail[0]), int(tail[1:3])


def parse_named_size_from_stem(stem: str) -> tuple[int, int] | None:
    """
    从文件名主体（不含后缀）解析“命名尺寸”对应的目标像素（100 DPI）。

    规则：取最后一个 '-' 之后的部分；为 3 位或 4 位纯数字时解析宽高（英寸），再 ×100 得像素。
    - 4 位：前两位为宽（inch），后两位为高（inch）
    - 3 位：第一位为宽，后两位为高
    """
    inches = _parse_named_inches_from_stem(stem)
    if inches is None:
        return None
    w_in, h_in = inches
    return (w_in * 100, h_in * 100)


def _pair_from_dpi_field(dpi_raw: Any) -> tuple[int, int] | None:
    if dpi_raw is None:
        return None
    if isinstance(dpi_raw, tuple) and len(dpi_raw) >= 2:
        try:
            x = int(round(float(dpi_raw[0])))
            y = int(round(float(dpi_raw[1])))
        except (TypeError, ValueError):
            return None
        return (x, y)
    try:
        v = int(round(float(dpi_raw)))
    except (TypeError, ValueError):
        return None
    return (v, v)


def _dpi_from_jfif_info(info: dict[str, Any]) -> tuple[int, int] | None:
    """JFIF 标记中的密度（单位 1=英寸 2=厘米）。"""
    if "jfif_density" not in info:
        return None
    jfif_density = info["jfif_density"]
    unit = info.get("jfif_unit")
    if not isinstance(jfif_density, tuple) or len(jfif_density) < 2:
        return None
    try:
        dx = float(jfif_density[0])
        dy = float(jfif_density[1])
    except (TypeError, ValueError):
        return None
    if unit == 1:
        return (round(dx), round(dy))
    if unit == 2:
        return (round(dx * 2.54), round(dy * 2.54))
    return None


def _rational_to_float(v: Any) -> float | None:
    if v is None:
        return None
    if hasattr(v, "numerator") and hasattr(v, "denominator"):
        d = float(v.denominator)
        if d == 0:
            return None
        return float(v.numerator) / d
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _dpi_from_exif(image: Image.Image) -> tuple[int, int] | None:
    """EXIF：XResolution / YResolution / ResolutionUnit（英寸或厘米）。"""
    try:
        exif = image.getexif()
    except Exception:
        return None
    if not exif:
        return None
    xf = _rational_to_float(exif.get(282))
    yf = _rational_to_float(exif.get(283))
    if xf is None or yf is None or xf <= 0 or yf <= 0:
        return None
    unit = exif.get(296)
    if unit == 3:
        return (round(xf * 2.54), round(yf * 2.54))
    if unit == 2 or unit is None:
        return (round(xf), round(yf))
    return None


def _dpi_from_png_phys_chunk(data: bytes) -> tuple[int, int] | None:
    """
    扫描 PNG 二进制中的 pHYs 块（像素/米），避开部分环境下未写入 info['dpi'] 的情况。
    unit=1：像素/米 → dpi = ppm × 0.0254
    """
    sig = b"\x89PNG\r\n\x1a\n"
    if len(data) < len(sig) + 12 or data[: len(sig)] != sig:
        return None
    offset = len(sig)
    while offset + 12 <= len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        ctype = data[offset + 4 : offset + 8]
        cdata_off = offset + 8
        if ctype == b"pHYs" and length >= 9 and cdata_off + length <= len(data):
            chunk = data[cdata_off : cdata_off + length]
            px = int.from_bytes(chunk[0:4], "big")
            py = int.from_bytes(chunk[4:8], "big")
            unit = chunk[8]
            if unit == 1:
                return (round(px * 0.0254), round(py * 0.0254))
            return None
        offset += 12 + length
    return None


def extract_dpi(
    image: Image.Image,
    *,
    raw_bytes: bytes | None = None,
) -> tuple[tuple[int, int] | None, str]:
    """
    优先从元数据读取 DPI；若无则尝试 JFIF、EXIF、PNG pHYs。
    均失败则返回 (None, \"未知\")。
    """
    info = image.info

    pair = _pair_from_dpi_field(info.get("dpi"))
    if pair:
        x, y = pair
        return (x, y), f"{x}×{y}"

    pair = _dpi_from_jfif_info(info)
    if pair:
        x, y = pair
        return (x, y), f"{x}×{y}（JFIF）"

    pair = _dpi_from_exif(image)
    if pair:
        x, y = pair
        return (x, y), f"{x}×{y}（EXIF）"

    if raw_bytes:
        pair = _dpi_from_png_phys_chunk(raw_bytes)
        if pair:
            x, y = pair
            return (x, y), f"{x}×{y}（PNG pHYs）"

    return None, "未知"


def is_match_100_dpi(dpi_pair: tuple[int, int] | None) -> bool:
    if dpi_pair is None:
        return False
    return dpi_pair[0] == 100 and dpi_pair[1] == 100


def style_status(value: str) -> str:
    """按匹配状态返回单元格样式。"""
    if value == "匹配":
        return "background-color: #d4edda; color: #155724;"
    if value == "不匹配":
        return "background-color: #f8d7da; color: #721c24;"
    return ""


def inspect_one(uploaded_name: str, raw_bytes: bytes) -> dict[str, Any]:
    """单张图片：目标尺寸、实际尺寸、DPI 文案、是否匹配。"""
    stem = uploaded_name.rsplit(".", 1)[0] if "." in uploaded_name else uploaded_name

    target = parse_named_size_from_stem(stem)
    if target is None:
        tw = th = None
        target_str = "—（无法从文件名解析）"
    else:
        tw, th = target
        target_str = f"{tw}×{th}"

    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.load()
        aw, ah = img.size
        actual_str = f"{aw}×{ah}"
        dpi_pair, dpi_display = extract_dpi(
            img,
            raw_bytes=raw_bytes,
        )
    except Exception as e:  # noqa: BLE001 — 展示用一行错误
        return {
            "图片名称": uploaded_name,
            "图片目标大小": target_str,
            "图片修改后尺寸": f"读取失败：{e}",
            "DPI值": "—",
            "匹配状态": "不匹配",
        }

    has_real_dpi = dpi_pair is not None

    if target is None:
        status = "不匹配"
    elif has_real_dpi and is_match_100_dpi(dpi_pair) and (
        (tw == aw and th == ah) or (tw == ah and th == aw)  # 核心修复：允许宽高互换（横竖图兼容）
    ):
        status = "匹配"
    else:
        status = "不匹配"

    return {
        "图片名称": uploaded_name,
        "图片目标大小": target_str,
        "图片修改后尺寸": actual_str,
        "DPI值": dpi_display,
        "匹配状态": status,
    }


def main() -> None:
    st.set_page_config(page_title="马克 33 号：图片参数体检专家", layout="wide")
    st.title("马克 33 号：图片参数体检专家")
    st.caption("批量校验文件名推算像素（100 DPI）与实际像素、DPI 是否一致。")

    files = st.file_uploader(
        "上传图片（可多选）",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="解析规则：取文件名（无后缀）最后一个「-」后的 3 或 4 位数字推算目标宽高（英寸）×100=像素。",
    )

    cbtn, ccap = st.columns([1, 4])
    with cbtn:
        st.button("重新体检", type="primary", help="在上传文件后点击可刷新结果")
    with ccap:
        st.caption("上传后会自动执行体检；也可点击「重新体检」刷新表格。")

    if not files:
        st.info("请上传至少一张 JPG/PNG 图片。")
        return

    total_count = len(files)
    
    # 新增：明确的上传完成提示
    st.info(f"📥 浏览器已将 {total_count} 张图片传输至系统内存，即将启动 4 线程极速体检引擎...")
    
    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()

    update_step = max(1, total_count // 50)
    rows = []
    completed = 0

    # 核心修复：使用 4 线程池并发处理图片解析
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 提交所有任务到线程池
        future_to_name = {
            executor.submit(inspect_one, getattr(f, "name", "未命名"), f.getvalue()): getattr(f, "name", "未命名")
            for f in files
        }

        # 获取完成的结果
        for future in as_completed(future_to_name):
            completed += 1
            name = future_to_name[future]
            try:
                row_result = future.result()
                rows.append(row_result)
            except Exception as e:
                rows.append({
                    "图片名称": name,
                    "图片目标大小": "—",
                    "图片修改后尺寸": f"处理异常: {e}",
                    "DPI值": "—",
                    "匹配状态": "不匹配"
                })

            # 实时更新 UI 进度条
            if completed == 1 or completed == total_count or completed % update_step == 0:
                progress_bar.progress(completed / total_count)
                status_placeholder.info(f"🚀 4 线程高速体检中：已完成 {completed}/{total_count} 张 (当前: {name}) ...")

    progress_bar.progress(1.0)
    progress_bar.empty()
    status_placeholder.empty()
    
    # 新增：上传与处理最终汇报
    st.success(f"✅ **上传与体检全部完成！** 系统已成功接收并极速分析了 **{total_count}** 张图片。")

    df = pd.DataFrame(rows)

    st.subheader("体检结果")
    styled_df = df.style.map(style_status, subset=["匹配状态"])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
        st.download_button(
            label="下载汇总表格（CSV）",
            data=csv_buf.getvalue(),
            file_name="图片参数体检汇总.csv",
            mime="text/csv",
        )
    with c2:
        xbuf = io.BytesIO()
        df.to_excel(xbuf, index=False, engine="openpyxl")
        st.download_button(
            label="下载汇总表格（Excel）",
            data=xbuf.getvalue(),
            file_name="图片参数体检汇总.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()
