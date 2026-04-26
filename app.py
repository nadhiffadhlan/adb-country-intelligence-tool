"""
Country Intelligence Tool — Streamlit Web App
===============================================
Built by: Nadhif Fadhlan Musyaffa — MSc Technopreneurship & Innovation, NTU Singapore
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
from reportlab.lib.units import cm as rcm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.platypus import Image as RLImage

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Country Intelligence Tool",
    page_icon="🌏",
    layout="wide"
)

st.title("🌏 Country Intelligence Tool")
st.markdown("AI-powered economic analysis using **World Bank data** + **LLM**")
st.divider()

# ─── API KEY FROM SECRETS ──────────────────────────────────────────────────────

API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")

# ─── COUNTRY NAME → CODE LOOKUP ───────────────────────────────────────────────

COUNTRY_MAP = {
    "afghanistan": "AF", "albania": "AL", "algeria": "DZ", "argentina": "AR",
    "australia": "AU", "austria": "AT", "bangladesh": "BD", "belgium": "BE",
    "brazil": "BR", "cambodia": "KH", "canada": "CA", "chile": "CL",
    "china": "CN", "colombia": "CO", "denmark": "DK", "egypt": "EG",
    "ethiopia": "ET", "finland": "FI", "france": "FR", "germany": "DE",
    "ghana": "GH", "greece": "GR", "hungary": "HU", "india": "IN",
    "indonesia": "ID", "iran": "IR", "iraq": "IQ", "ireland": "IE",
    "israel": "IL", "italy": "IT", "japan": "JP", "jordan": "JO",
    "kenya": "KE", "south korea": "KR", "korea": "KR", "kuwait": "KW",
    "laos": "LA", "malaysia": "MY", "mexico": "MX", "mongolia": "MN",
    "morocco": "MA", "myanmar": "MM", "nepal": "NP", "netherlands": "NL",
    "the netherlands": "NL", "new zealand": "NZ", "nigeria": "NG",
    "norway": "NO", "pakistan": "PK", "peru": "PE", "philippines": "PH",
    "poland": "PL", "portugal": "PT", "romania": "RO", "russia": "RU",
    "saudi arabia": "SA", "singapore": "SG", "south africa": "ZA",
    "spain": "ES", "sri lanka": "LK", "sweden": "SE", "switzerland": "CH",
    "taiwan": "TW", "thailand": "TH", "turkey": "TR", "ukraine": "UA",
    "united arab emirates": "AE", "uae": "AE", "united kingdom": "GB",
    "uk": "GB", "united states": "US", "usa": "US", "america": "US",
    "uzbekistan": "UZ", "vietnam": "VN", "viet nam": "VN", "zambia": "ZM",
    "zimbabwe": "ZW",
}

def cari_kode_negara(nama):
    key = nama.strip().lower()
    if key in COUNTRY_MAP:
        return COUNTRY_MAP[key]
    for k, v in COUNTRY_MAP.items():
        if key in k or k in key:
            return v
    return None

# ─── CONFIG ────────────────────────────────────────────────────────────────────

MODELS = [
    "google/gemma-3-4b-it:free",
    "mistralai/devstral-small:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]

COUNTRY_COLORS = ["steelblue", "crimson", "seagreen", "darkorange", "purple"]

# ─── FUNCTIONS ─────────────────────────────────────────────────────────────────

def ambil_data(indikator, kode_negara, tahun_mulai=1970, tahun_selesai=2025):
    url = (
        f"https://api.worldbank.org/v2/country/{kode_negara}/indicator/{indikator}"
        f"?date={tahun_mulai}:{tahun_selesai}&format=json&per_page=100"
    )
    try:
        with urllib.request.urlopen(url) as r:
            data = json.loads(r.read())
    except:
        return pd.DataFrame(columns=["Tahun", "Nilai"])

    records = []
    if len(data) > 1 and isinstance(data[1], list):
        for item in data[1]:
            if item["value"] is not None:
                records.append({"Tahun": int(item["date"]), "Nilai": item["value"]})

    return pd.DataFrame(records).sort_values("Tahun")


def tanya_ai(prompt):
    if not API_KEY:
        return "API key not available."

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
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

    return "All models failed. Please try again later."


def buat_grafik_single(gdp, pendidikan, nama_negara):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"Economic Profile: {nama_negara}", fontsize=13, fontweight="bold")

    axes[0].plot(gdp["Tahun"], gdp["Nilai"] / 1e9,
                 color="steelblue", marker="o", markersize=3)
    axes[0].set_title("GDP (Billion USD)")
    axes[0].set_xlabel("Year")
    axes[0].grid(True, alpha=0.3)

    if not pendidikan.empty:
        axes[1].plot(pendidikan["Tahun"], pendidikan["Nilai"],
                     color="purple", marker="o", markersize=3)
        axes[1].set_title("Education Spending (% GDP)")
        axes[1].set_xlabel("Year")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "Data not available",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("Education Spending (% GDP)")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf


def buat_grafik_comparison(negara_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Regional Economic Comparison", fontsize=14, fontweight="bold")

    for i, nd in enumerate(negara_data):
        color = COUNTRY_COLORS[i % len(COUNTRY_COLORS)]
        if not nd["gdp"].empty:
            axes[0].plot(nd["gdp"]["Tahun"], nd["gdp"]["Nilai"] / 1e9,
                         color=color, marker="o", markersize=3,
                         linewidth=2, label=nd["nama"])
        if not nd["pendidikan"].empty:
            axes[1].plot(nd["pendidikan"]["Tahun"], nd["pendidikan"]["Nilai"],
                         color=color, marker="o", markersize=3,
                         linewidth=2, label=nd["nama"])

    for ax, title, ylabel in zip(
        axes,
        ["GDP (Billion USD)", "Education Spending (% GDP)"],
        ["Year", "Year"]
    ):
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf


def generate_pdf_single(nama_negara, gdp_terbaru, gdp_growth, gdp_tahun_awal, edu_val, analisis, chart_buf):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*rcm, leftMargin=2*rcm,
                            topMargin=2*rcm, bottomMargin=2*rcm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("T", parent=styles["Title"], fontSize=20,
                                  textColor=colors.HexColor("#1a3a6b"), spaceAfter=4)
    sub_style   = ParagraphStyle("S", parent=styles["Normal"], fontSize=9,
                                  textColor=colors.HexColor("#666666"), spaceAfter=2)
    sec_style   = ParagraphStyle("H", parent=styles["Heading2"], fontSize=13,
                                  textColor=colors.HexColor("#1a3a6b"), spaceBefore=12, spaceAfter=6)
    body_style  = ParagraphStyle("B", parent=styles["Normal"], fontSize=10, leading=15, spaceAfter=6)

    story = []
    story.append(Paragraph(f"Country Intelligence Brief: {nama_negara.upper()}", title_style))
    story.append(Paragraph(f"Generated by Country Intelligence Tool | {datetime.date.today().strftime('%d %B %Y')}", sub_style))
    story.append(Paragraph("Built by Nadhif Fadhlan Musyaffa — MSc Technopreneurship & Innovation, NTU Singapore", sub_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a3a6b"), spaceAfter=12))

    story.append(Paragraph("Key Economic Indicators", sec_style))
    tbl = Table([
        ["Indicator", "Value"],
        ["GDP (Latest)", f"${gdp_terbaru:.1f} Billion USD"],
        ["GDP Growth", f"{gdp_growth:.1f}% since {gdp_tahun_awal}"],
        ["Education Spending", edu_val],
    ], colWidths=[8*rcm, 8*rcm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#1a3a6b")),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f4ff"), colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("PADDING",        (0, 0), (-1, -1), 8),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Economic Trends", sec_style))
    story.append(RLImage(chart_buf, width=16*rcm, height=5*rcm))
    story.append(Spacer(1, 12))

    story.append(Paragraph("AI-Generated Country Brief", sec_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=8))
    for para in analisis.replace("**", "").replace("*", "").split("\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), body_style))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=6))
    story.append(Paragraph("Data: World Bank Open Data | AI: OpenRouter LLM | github.com/nadhiffadhlan/adb-country-intelligence-tool", sub_style))

    doc.build(story)
    buf.seek(0)
    return buf


def generate_pdf_comparison(negara_data, analisis_regional, chart_buf):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*rcm, leftMargin=2*rcm,
                            topMargin=2*rcm, bottomMargin=2*rcm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("T", parent=styles["Title"], fontSize=18,
                                  textColor=colors.HexColor("#1a3a6b"), spaceAfter=4)
    sub_style   = ParagraphStyle("S", parent=styles["Normal"], fontSize=9,
                                  textColor=colors.HexColor("#666666"), spaceAfter=2)
    sec_style   = ParagraphStyle("H", parent=styles["Heading2"], fontSize=13,
                                  textColor=colors.HexColor("#1a3a6b"), spaceBefore=12, spaceAfter=6)
    body_style  = ParagraphStyle("B", parent=styles["Normal"], fontSize=10, leading=15, spaceAfter=6)

    story = []
    negara_names = " vs ".join([nd["nama"] for nd in negara_data])
    story.append(Paragraph(f"Regional Comparison: {negara_names}", title_style))
    story.append(Paragraph(f"Generated by Country Intelligence Tool | {datetime.date.today().strftime('%d %B %Y')}", sub_style))
    story.append(Paragraph("Built by Nadhif Fadhlan Musyaffa — MSc Technopreneurship & Innovation, NTU Singapore", sub_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a3a6b"), spaceAfter=12))

    story.append(Paragraph("Key Economic Indicators Comparison", sec_style))
    rows = [["Country", "GDP (Latest)", "GDP Growth", "Education Spending"]]
    for nd in negara_data:
        gdp = nd["gdp"]
        edu = nd["pendidikan"]
        if not gdp.empty:
            gdp_terbaru = gdp.iloc[-1]["Nilai"] / 1e9
            gdp_growth  = ((gdp_terbaru - gdp.iloc[0]["Nilai"]/1e9) / (gdp.iloc[0]["Nilai"]/1e9)) * 100
            rows.append([nd["nama"], f"${gdp_terbaru:.1f}B", f"{gdp_growth:.1f}%",
                         f"{edu.iloc[-1]['Nilai']:.1f}% GDP" if not edu.empty else "N/A"])

    tbl = Table(rows, colWidths=[4*rcm, 4*rcm, 4*rcm, 4*rcm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#1a3a6b")),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f4ff"), colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("PADDING",        (0, 0), (-1, -1), 7),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Regional Economic Trends", sec_style))
    story.append(RLImage(chart_buf, width=16*rcm, height=6*rcm))
    story.append(Spacer(1, 12))

    story.append(Paragraph("AI Regional Analysis", sec_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=8))
    for para in analisis_regional.replace("**", "").replace("*", "").split("\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), body_style))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=6))
    story.append(Paragraph("Data: World Bank Open Data | AI: OpenRouter LLM | github.com/nadhiffadhlan/adb-country-intelligence-tool", sub_style))

    doc.build(story)
    buf.seek(0)
    return buf


# ─── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🌏 Country Intelligence Tool")
    st.divider()
    st.markdown("Nadhif Fadhlan Musyaffa")
    st.markdown("[GitHub Repo](https://github.com/nadhiffadhlan/adb-country-intelligence-tool)")

# ─── MODE SELECTOR ─────────────────────────────────────────────────────────────

mode = st.radio(
    "Analysis Mode",
    ["🔍 Single Country", "🌐 Multi-Country Comparison"],
    horizontal=True
)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: SINGLE COUNTRY
# ══════════════════════════════════════════════════════════════════════════════

if mode == "🔍 Single Country":
    st.subheader("📍 Select Country")
    nama_negara = st.text_input("Country Name", placeholder="e.g. Indonesia, Japan, United States")
    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    if analyze_btn:
        if not nama_negara:
            st.error("Please enter a country name.")
        else:
            kode = cari_kode_negara(nama_negara)
            if not kode:
                st.error(f"Country '{nama_negara}' not recognized. Try another name, e.g. 'Indonesia', 'Japan', 'Germany'.")
            else:
                st.caption(f"🔎 Detected country code: **{kode}**")
                with st.spinner(f"⏳ Fetching data for {nama_negara}..."):
                    gdp        = ambil_data("NY.GDP.MKTP.CD", kode)
                    pendidikan = ambil_data("SE.XPD.TOTL.GD.ZS", kode)

                if gdp.empty:
                    st.error(f"No GDP data found for '{nama_negara}'.")
                else:
                    gdp_terbaru    = gdp.iloc[-1]["Nilai"] / 1e9
                    gdp_terlama    = gdp.iloc[0]["Nilai"] / 1e9
                    gdp_growth     = ((gdp_terbaru - gdp_terlama) / gdp_terlama) * 100
                    gdp_tahun_awal = int(gdp.iloc[0]["Tahun"])
                    edu_val        = f"{pendidikan.iloc[-1]['Nilai']:.1f}% of GDP" if not pendidikan.empty else "N/A"

                    st.subheader(f"📊 Economic Data: {nama_negara.upper()}")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Latest GDP", f"${gdp_terbaru:.1f}B USD")
                    m2.metric("GDP Growth", f"{gdp_growth:.1f}%", f"since {gdp_tahun_awal}")
                    m3.metric("Education Spending", edu_val)

                    chart_buf = buat_grafik_single(gdp, pendidikan, nama_negara)
                    st.image(chart_buf, use_column_width=True)

                    st.subheader("🤖 AI Country Brief")
                    with st.spinner("Generating analysis..."):
                        analisis = tanya_ai(f"""
                        You are a senior economist at ADB writing a formal report.
                        Write a concise analysis (1 paragraph) about {nama_negara}:
                        - Latest GDP: ${gdp_terbaru:.1f} Billion USD
                        - GDP growth since {gdp_tahun_awal}: {gdp_growth:.1f}%
                        - Education spending: {edu_val}
                        Focus: economic trajectory, key challenges, ADB investment recommendations.
                        Write in formal but accessible English.
                        Do NOT ask follow-up questions. Do NOT offer further assistance. End with a concluding sentence only.
                        """)
                    st.info(analisis)

                    st.divider()
                    with st.spinner("Preparing PDF..."):
                        pdf_buf = generate_pdf_single(
                            nama_negara, gdp_terbaru, gdp_growth,
                            gdp_tahun_awal, edu_val, analisis,
                            buat_grafik_single(gdp, pendidikan, nama_negara)
                        )
                    st.download_button(
                        label="⬇️ Download PDF Country Brief",
                        data=pdf_buf,
                        file_name=f"Brief_{nama_negara.replace(' ','_')}_{datetime.date.today()}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary"
                    )
                    st.caption("Data: World Bank Open Data | AI: OpenRouter LLM")

# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: MULTI-COUNTRY COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

else:
    st.subheader("🌐 Multi-Country Comparison")
    st.caption("Enter 2–5 countries in English for comparison.")

    jumlah = st.slider("Number of countries", min_value=2, max_value=5, value=3)
    cols   = st.columns(jumlah)
    nama_inputs = []
    for i, col in enumerate(cols):
        with col:
            nama = st.text_input(f"Country {i+1}", placeholder="e.g. Indonesia", key=f"nama_{i}")
            nama_inputs.append(nama)

    compare_btn = st.button("🌐 Compare Now", type="primary", use_container_width=True)

    if compare_btn:
        valid_inputs = [n for n in nama_inputs if n.strip()]
        if len(valid_inputs) < 2:
            st.error("Please enter at least 2 countries.")
        else:
            negara_data = []
            for nama in valid_inputs:
                kode = cari_kode_negara(nama)
                if not kode:
                    st.warning(f"⚠️ '{nama}' not recognized, skipping.")
                    continue
                st.caption(f"🔎 {nama} → **{kode}**")
                with st.spinner(f"⏳ Fetching data for {nama}..."):
                    gdp = ambil_data("NY.GDP.MKTP.CD", kode)
                    edu = ambil_data("SE.XPD.TOTL.GD.ZS", kode)
                if not gdp.empty:
                    negara_data.append({"nama": nama, "gdp": gdp, "pendidikan": edu})
                else:
                    st.warning(f"⚠️ No data found for {nama}, skipping.")

            if len(negara_data) < 2:
                st.error("At least 2 countries must have valid data.")
            else:
                st.subheader("📊 Economic Indicators Comparison")
                rows = []
                for nd in negara_data:
                    gdp = nd["gdp"]
                    edu = nd["pendidikan"]
                    gdp_terbaru = gdp.iloc[-1]["Nilai"] / 1e9
                    gdp_growth  = ((gdp_terbaru - gdp.iloc[0]["Nilai"]/1e9) / (gdp.iloc[0]["Nilai"]/1e9)) * 100
                    rows.append({
                        "Country": nd["nama"],
                        "Latest GDP (B USD)": f"${gdp_terbaru:.1f}",
                        "GDP Growth": f"{gdp_growth:.1f}%",
                        "Education Spending": f"{edu.iloc[-1]['Nilai']:.1f}%" if not edu.empty else "N/A"
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                st.subheader("📈 Overlay Chart")
                chart_buf = buat_grafik_comparison(negara_data)
                st.image(chart_buf, use_column_width=True)

                st.subheader("🤖 AI Regional Analysis")
                negara_summary = ""
                for nd in negara_data:
                    gdp = nd["gdp"]
                    edu = nd["pendidikan"]
                    gdp_terbaru = gdp.iloc[-1]["Nilai"] / 1e9
                    gdp_growth  = ((gdp_terbaru - gdp.iloc[0]["Nilai"]/1e9) / (gdp.iloc[0]["Nilai"]/1e9)) * 100
                    edu_str     = f"{edu.iloc[-1]['Nilai']:.1f}%" if not edu.empty else "N/A"
                    negara_summary += f"\n- {nd['nama']}: GDP ${gdp_terbaru:.1f}B, growth {gdp_growth:.1f}%, education {edu_str}"

                with st.spinner("Generating regional analysis..."):
                    analisis_regional = tanya_ai(f"""
                    You are a senior regional economist at ADB writing a formal report.
                    Compare the following economies in 2 paragraphs:
                    {negara_summary}
                    Focus: which is strongest, which is growing fastest,
                    and what are the implications for ADB's investment strategy in the region.
                    Write in formal but accessible English.
                    Do NOT ask follow-up questions. Do NOT offer further assistance. End with a concluding sentence only.
                    """)
                st.info(analisis_regional)

                st.divider()
                with st.spinner("Preparing PDF..."):
                    pdf_buf = generate_pdf_comparison(
                        negara_data, analisis_regional,
                        buat_grafik_comparison(negara_data)
                    )
                negara_str = "_vs_".join([nd["nama"].replace(" ", "") for nd in negara_data])
                st.download_button(
                    label="⬇️ Download PDF Regional Report",
                    data=pdf_buf,
                    file_name=f"Regional_{negara_str}_{datetime.date.today()}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )
                st.caption("Data: World Bank Open Data | AI: OpenRouter LLM")
