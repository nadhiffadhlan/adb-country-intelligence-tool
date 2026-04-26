"""
ADB Country Intelligence Tool — Streamlit Web App
===================================================
Built by: Nadhif Fadhlan — MSc Technopreneurship & Innovation, NTU Singapore
"""

import urllib.request
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ADB Country Intelligence Tool",
    page_icon="🌏",
    layout="wide"
)

# ─── HEADER ────────────────────────────────────────────────────────────────────

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
                records.append({
                    "Tahun": int(item["date"]),
                    "Nilai": item["value"]
                })

    return pd.DataFrame(records).sort_values("Tahun")


def tanya_ai(prompt, api_key):
    if not api_key:
        return "⚠️ API key kosong. Masukkan OpenRouter API key di sidebar."

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

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
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
            st.error(f"Data GDP untuk '{kode_negara}' tidak ditemukan. Cek kode negaranya ya!")
        else:
            # Stats
            gdp_terbaru = gdp.iloc[-1]["Nilai"] / 1e9
            gdp_terlama = gdp.iloc[0]["Nilai"] / 1e9
            gdp_growth  = ((gdp_terbaru - gdp_terlama) / gdp_terlama) * 100

            st.subheader(f"📊 Data Ekonomi: {nama_negara.upper()}")

            m1, m2, m3 = st.columns(3)
            m1.metric("GDP Terbaru", f"${gdp_terbaru:.1f}B USD")
            m2.metric("GDP Growth", f"{gdp_growth:.1f}%", f"sejak {int(gdp.iloc[0]['Tahun'])}")
            if not pendidikan.empty:
                m3.metric("Education Spending", f"{pendidikan.iloc[-1]['Nilai']:.1f}% GDP")
            else:
                m3.metric("Education Spending", "N/A")

            # Charts
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
            st.pyplot(fig)

            # AI Brief
            st.subheader("🤖 AI Country Brief")
            with st.spinner("AI lagi nulis analisis..."):
                edu_val = f"{pendidikan.iloc[-1]['Nilai']:.1f}%" if not pendidikan.empty else "N/A"
                prompt = f"""
                Kamu adalah senior economist di ADB.
                Tulis analisis singkat (1 paragraf) tentang {nama_negara}:
                - GDP terbaru: ${gdp_terbaru:.1f} Miliar USD
                - GDP growth sejak {int(gdp.iloc[0]['Tahun'])}: {gdp_growth:.1f}%
                - Education spending: {edu_val}
                Fokus: trajectory ekonomi, tantangan utama, rekomendasi investasi ADB.
                Bahasa Indonesia, formal tapi accessible.
                """
                analisis = tanya_ai(prompt, api_key)

            st.info(analisis)

            st.caption("Data: World Bank Open Data | AI: OpenRouter LLM")
