"""
ADB Country Intelligence Tool
==============================
AI-powered economic analysis tool that pulls real-time data
from World Bank API and generates country briefs using LLM.

Built by: Nadhif Fadhlan — MSc Technopreneurship & Innovation, NTU Singapore

Usage:
    Set your OpenRouter API key as environment variable:
        export OPENROUTER_API_KEY="sk-or-..."
    Then run:
        python main.py
"""

import urllib.request
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# ─── CONFIG ────────────────────────────────────────────────────────────────────

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

MODELS = [
    "google/gemma-3-4b-it:free",
    "mistralai/devstral-small:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]

# ─── FUNCTIONS ─────────────────────────────────────────────────────────────────

def ambil_data(indikator, negara, tahun_mulai=1970, tahun_selesai=2025):
    """Fetch indicator data from World Bank API."""
    url = (
        f"https://api.worldbank.org/v2/country/{negara}/indicator/{indikator}"
        f"?date={tahun_mulai}:{tahun_selesai}&format=json&per_page=100"
    )
    try:
        with urllib.request.urlopen(url) as r:
            data = json.loads(r.read())
    except Exception as e:
        print(f"⚠️  Gagal mengambil data dari World Bank: {e}")
        return pd.DataFrame(columns=["Tahun", "Nilai"])

    records = []
    if len(data) > 1 and isinstance(data[1], list):
        for item in data[1]:
            if item["value"] is not None:
                records.append({
                    "Tahun": int(item["date"]),
                    "Nilai": item["value"]
                })
    else:
        print(f"⚠️  Tidak ada data untuk {negara} - indikator {indikator}")
        return pd.DataFrame(columns=["Tahun", "Nilai"])

    return pd.DataFrame(records).sort_values("Tahun")


def tanya_ai(prompt):
    """Send prompt to OpenRouter, try each model in MODELS until one works."""
    if not API_KEY:
        return "⚠️  API key tidak ditemukan. Set OPENROUTER_API_KEY di environment variable."

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
            print(f"✅ Pakai model: {model}")
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"❌ {model} gagal: {e}")
            continue

    return "Semua model gagal. Coba lagi nanti."


def analisis_negara(nama_negara, kode_negara):
    """Full pipeline: fetch data, plot charts, generate AI brief."""
    print(f"\n⏳ Mengambil data {nama_negara} dari World Bank...\n")

    gdp        = ambil_data("NY.GDP.MKTP.CD", kode_negara)
    pendidikan = ambil_data("SE.XPD.TOTL.GD.ZS", kode_negara)

    if gdp.empty:
        print(f"⚠️  Data GDP {nama_negara} tidak tersedia. Skip.\n")
        return

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    fig.suptitle(f"Economic Profile: {nama_negara}", fontsize=14, fontweight="bold")

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
    plt.show()

    # Summary stats
    gdp_terbaru = gdp.iloc[-1]["Nilai"] / 1e9
    gdp_terlama = gdp.iloc[0]["Nilai"] / 1e9
    gdp_growth  = ((gdp_terbaru - gdp_terlama) / gdp_terlama) * 100

    print(f"\n📊 RINGKASAN DATA {nama_negara.upper()}")
    print(f"GDP terbaru:  ${gdp_terbaru:.1f} Miliar")
    print(f"GDP growth:   {gdp_growth:.1f}% sejak {gdp.iloc[0]['Tahun']}")
    if not pendidikan.empty:
        print(f"Education spending: {pendidikan.iloc[-1]['Nilai']:.1f}% of GDP")

    # AI brief
    print(f"\n🤖 AI lagi nulis analisis {nama_negara}...\n")

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

    analisis = tanya_ai(prompt)
    print("=" * 60)
    print(f"📋 COUNTRY BRIEF: {nama_negara.upper()}")
    print("=" * 60)
    print(analisis)
    print("=" * 60)


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("       ADB COUNTRY INTELLIGENCE TOOL 🌏")
    print("=" * 60)

    jumlah = int(input("\nMau analisis berapa negara? "))
    negara_list = []

    for i in range(jumlah):
        nama = input(f"\nNama negara {i+1}: ")
        kode = input(f"Kode negara {i+1} (2 huruf, contoh: ID, JP, SG): ").upper()
        negara_list.append((nama, kode))

    for nama_negara, kode_negara in negara_list:
        analisis_negara(nama_negara, kode_negara)

    print("\n✅ Selesai! Semua negara sudah dianalisis.")


if __name__ == "__main__":
    main()
