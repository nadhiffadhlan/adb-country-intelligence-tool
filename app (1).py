"""
ADB Country Intelligence Tool — Streamlit Web App
===================================================
Built by: Nadhif Fadhlan — MSc Technopreneurship & Innovation, NTU Singapore
"""

import urllib.request
import json
import io
import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.platypus import Image as RLImage

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ADB Country Intelligence Tool",
    page_icon="🌏",
    layout="wide"
)

st.title("🌏 ADB Country Intelligence Tool")
st.markdown("AI-powered economic analysis using **World Bank data** + **LLM**")
st.divider()

# ─── CONFIG ────────────────────────────────────────────────────────────────────

MODELS = [
    "google/gemma-3-4b-it:free",
    "mistralai/devstral-small:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]

# ─── FUNCTIONS ─────────────────────────────────────────────────────────────────

def ambil_data(indikator, negara, tahun_mulai=1970, tahun_selesai=2025):
    url = (
        f"https://api.worldbank.org/v2/country/{negara}/indicator/{indikator}"
        f"?date={tahun_mulai}:{tahun_selesai}&format=json&per_page=100"
    )
    try:
        with urllib.request.urlopen(url) as r:
            data = json.loads(r.read())
    except Exception as e:
        st.warning(f"Gagal ambil data World Bank: {e}")
        return pd.DataFrame(columns=["Tahun", "Nilai"])

    records = []
    if len(data) > 1 and isinstance(data[1], list):
        for item in data[1]:
            if item["value"] is not None:
                records.append({"Tahun": int(item["date"]), "Nilai": item["value"]})

    return pd.DataFrame(records).sort_values("Tahun")


def tanya_ai(prompt, api_key):
    if not api_key:
        return "API key kosong. Masukkan OpenRouter API key di sidebar."

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    for model in MODELS:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req) as r:
                result = json.loads(r.read())
            return result["choices"][0]["message"]["content"]
        except:
            continue

    return "Semua model gagal. Coba lagi nanti."


def buat_grafik(gdp, pendidikan, nama_negara):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"Economic Profile: {nama_negara}", fontsize=13, fontweight="bold")

    axes[0].plot(gdp["Tahun"], gdp["Nilai"] / 1e9,
                 color="steelblue", marker="o", markersize=3)
    axes[0].set_title("GDP (Miliar USD)")
    axes[0].set_xlabel("Tahun")
    axes[0].grid(True, alpha=0.3)

    if not pendidikan.empty:
        axes[1].plot(pendidikan["Tahun"], pendidikan["Nilai"],
                     color="purple", marker="o", markersize=3)
        axes[1].set_title("Education Spending (% GDP)")
        axes[1].set_xlabel("Tahun")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "Data tidak tersedia",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("Education Spending (% GDP)")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf


def generate_pdf(nama_negara, gdp_terbaru, gdp_growth, gdp_tahun_awal, edu_val, analisis, chart_buf):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("T", parent=styles["Title"], fontSize=20,
                                  textColor=colors.HexColor("#1a3a6b"), spaceAfter=4)
    sub_style   = ParagraphStyle("S", parent=styles["Normal"], fontSize=9,
                                  textColor=colors.HexColor("#666666"), spaceAfter=2)
    sec_style   = ParagraphStyle("H", parent=styles["Heading2"], fontSize=13,
                                  textColor=colors.HexColor("#1a3a6b"), spaceBefore=12, spaceAfter=6)
    body_style  = ParagraphStyle("B", parent=styles["Normal"], fontSize=10, leading=15, spaceAfter=6)

    story = []

    # Header
    story.append(Paragraph(f"Country Intelligence Brief: {nama_negara.upper()}", title_style))
    story.append(Paragraph(
        f"Generated by ADB Country Intelligence Tool | {datetime.date.today().strftime('%d %B %Y')}",
        sub_style
    ))
    story.append(Paragraph(
        "Built by Nadhif Fadhlan — MSc Technopreneurship & Innovation, NTU Singapore",
        sub_style
    ))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor("#1a3a6b"), spaceAfter=12))

    # Stats table
    story.append(Paragraph("Key Economic Indicators", sec_style))
    table_data = [
        ["Indicator", "Value"],
        ["GDP (Latest)", f"${gdp_terbaru:.1f} Billion USD"],
        ["GDP Growth", f"{gdp_growth:.1f}% since {gdp_tahun_awal}"],
        ["Education Spending", edu_val],
    ]
    tbl = Table(table_data, colWidths=[8*cm, 8*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#1a3a6b")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f0f4ff"), colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("PADDING",       (0, 0), (-1, -1), 8),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 12))

    # Chart
    story.append(Paragraph("Economic Trends", sec_style))
    story.append(RLImage(chart_buf, width=16*cm, height=5*cm))
    story.append(Spacer(1, 12))

    # AI Brief
    story.append(Paragraph("AI-Generated Country Brief", sec_style))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#cccccc"), spaceAfter=8))
    clean = analisis.replace("**", "").replace("*", "")
    for para in clean.split("\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), body_style))

    # Footer
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#cccccc"), spaceAfter=6))
    story.append(Paragraph(
        "Data: World Bank Open Data | AI: OpenRouter LLM | "
        "github.com/nadhiffadhlan/adb-country-intelligence-tool",
        sub_style
    ))

    doc.build(story)
    buf.seek(0)
    return buf


# ─── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "OpenRouter API Key", type="password",
        placeholder="sk-or-...",
        help="Dapatkan key gratis di openrouter.ai/keys"
    )
    st.caption("🔒 Key kamu tidak disimpan — hanya dipakai saat sesi ini.")
    st.divider()
    st.markdown("**Built by**")
    st.markdown("Nadhif Fadhlan")
    st.markdown("MSc Technopreneurship & Innovation, NTU Singapore")
    st.markdown("[GitHub Repo](https://github.com/nadhiffadhlan/adb-country-intelligence-tool)")

# ─── INPUT ─────────────────────────────────────────────────────────────────────

st.subheader("📍 Pilih Negara")
col1, col2 = st.columns(2)
with col1:
    nama_negara = st.text_input("Nama Negara", placeholder="contoh: Indonesia")
with col2:
    kode_negara = st.text_input("Kode Negara (2 huruf)", placeholder="contoh: ID").upper()

analyze_btn = st.button("🔍 Analisis Sekarang", type="primary", use_container_width=True)

# ─── ANALYSIS ──────────────────────────────────────────────────────────────────

if analyze_btn:
    if not nama_negara or not kode_negara:
        st.error("Isi nama negara dan kode negara dulu ya!")
    else:
        with st.spinner(f"⏳ Mengambil data {nama_negara} dari World Bank..."):
            gdp        = ambil_data("NY.GDP.MKTP.CD", kode_negara)
            pendidikan = ambil_data("SE.XPD.TOTL.GD.ZS", kode_negara)

        if gdp.empty:
            st.error(f"Data GDP untuk '{kode_negara}' tidak ditemukan. Cek kode negaranya!")
        else:
            gdp_terbaru    = gdp.iloc[-1]["Nilai"] / 1e9
            gdp_terlama    = gdp.iloc[0]["Nilai"] / 1e9
            gdp_growth     = ((gdp_terbaru - gdp_terlama) / gdp_terlama) * 100
            gdp_tahun_awal = int(gdp.iloc[0]["Tahun"])
            edu_val        = f"{pendidikan.iloc[-1]['Nilai']:.1f}% of GDP" if not pendidikan.empty else "N/A"

            # Metrics
            st.subheader(f"📊 Data Ekonomi: {nama_negara.upper()}")
            m1, m2, m3 = st.columns(3)
            m1.metric("GDP Terbaru", f"${gdp_terbaru:.1f}B USD")
            m2.metric("GDP Growth", f"{gdp_growth:.1f}%", f"sejak {gdp_tahun_awal}")
            m3.metric("Education Spending", edu_val)

            # Chart
            chart_buf = buat_grafik(gdp, pendidikan, nama_negara)
            st.image(chart_buf, use_column_width=True)

            # AI Brief
            st.subheader("🤖 AI Country Brief")
            with st.spinner("AI lagi nulis analisis..."):
                prompt = f"""
                Kamu adalah senior economist di ADB.
                Tulis analisis singkat (1 paragraf) tentang {nama_negara}:
                - GDP terbaru: ${gdp_terbaru:.1f} Miliar USD
                - GDP growth sejak {gdp_tahun_awal}: {gdp_growth:.1f}%
                - Education spending: {edu_val}
                Fokus: trajectory ekonomi, tantangan utama, rekomendasi investasi ADB.
                Bahasa Indonesia, formal tapi accessible.
                """
                analisis = tanya_ai(prompt, api_key)
            st.info(analisis)

            # PDF Export
            st.divider()
            with st.spinner("Menyiapkan PDF..."):
                chart_buf_pdf = buat_grafik(gdp, pendidikan, nama_negara)
                pdf_buf = generate_pdf(
                    nama_negara, gdp_terbaru, gdp_growth,
                    gdp_tahun_awal, edu_val, analisis, chart_buf_pdf
                )

            st.download_button(
                label="⬇️ Download PDF Country Brief",
                data=pdf_buf,
                file_name=f"ADB_Brief_{nama_negara.replace(' ', '_')}_{datetime.date.today()}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )

            st.caption("Data: World Bank Open Data | AI: OpenRouter LLM")
