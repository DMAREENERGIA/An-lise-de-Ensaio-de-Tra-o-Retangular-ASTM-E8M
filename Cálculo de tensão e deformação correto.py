import os
import sys
import tempfile
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from fpdf import FPDF


def processar_ensaio(dados_curva, w0, t0, L0, wf, tf, Lf, E_gpa, offset_pct, ponto_zero_pct):
    # Área inicial e final (mm²)
    A0 = w0 * t0
    Af = wf * tf

    # Conversão do Módulo de Elasticidade (GPa para MPa / N/mm²)
    E_mpa = E_gpa * 1000.0  

    alongamento_pct = ((Lf - L0) / L0) * 100.0
    estriccao_pct = ((A0 - Af) / A0) * 100.0

    historico_deformacoes = []
    historico_tensoes = []

    for carga_kn, desloc_mm in dados_curva:
        carga_n = carga_kn * 1000.0
        tensao_n_mm2 = carga_n / A0
        deformacao_pct = (desloc_mm / L0) * 100.0

        historico_tensoes.append(tensao_n_mm2)
        historico_deformacoes.append(deformacao_pct)

    arr_eps = np.array(historico_deformacoes)  # Deformação em %
    arr_sig = np.array(historico_tensoes)      # Tensão em MPa (N/mm²)

    idx_max = np.argmax(arr_sig)
    sigma_max = arr_sig[idx_max]
    carga_max_n = sigma_max * A0
    desloc_max = arr_eps[idx_max] * L0 / 100.0

    # Deslocamento Total da Reta de Offset (Ponto Zero + Offset)
    offset_total = ponto_zero_pct + offset_pct

    # Reta de Offset: y = E * (eps_abs - offset_total_abs)
    y_reta_offset = E_mpa * ((arr_eps - offset_total) / 100.0)
    diferenca = arr_sig - y_reta_offset

    # Localiza cruzamento da curva de tração com a reta de offset (mudança de sinal)
    cruzamentos = np.where(np.diff(np.sign(diferenca)))[0]

    if len(cruzamentos) > 0:
        i = cruzamentos[0]
        # Interpolação linear precisa para encontrar a interseção exata
        d1 = diferenca[i]
        d2 = diferenca[i + 1]
        t = -d1 / (d2 - d1) if d2 != d1 else 0.0

        deformacao_escoamento = arr_eps[i] + t * (arr_eps[i + 1] - arr_eps[i])
        tensao_escoamento = arr_sig[i] + t * (arr_sig[i + 1] - arr_sig[i])
    else:
        tensao_escoamento = sigma_max
        deformacao_escoamento = arr_eps[idx_max]

    carga_escoamento_n = tensao_escoamento * A0
    razao_elastica = tensao_escoamento / sigma_max if sigma_max > 0 else 0

    # --- Construção do Gráfico com Tamanho DOBRADO (28x16) ---
    fig, ax = plt.subplots(figsize=(28, 16))
    
    # Aumentada a espessura da linha
    ax.plot(arr_eps, arr_sig, 'b-', label='Curva Tensão-Deformação (Aço)', linewidth=4.0)

    # Limite superior de tensão para limitar as retas e o eixo Y
    y_lim_max = sigma_max * 1.15 if sigma_max > 0 else 600.0

    # 1. Reta da Região Elástica (Ponto Zero Deslocável)
    x_lin_max = ponto_zero_pct + (y_lim_max / E_mpa) * 100.0
    x_lin = np.linspace(ponto_zero_pct, x_lin_max, 100)
    y_lin = E_mpa * ((x_lin - ponto_zero_pct) / 100.0)
    ax.plot(x_lin, y_lin, 'k--', alpha=0.6, linewidth=2.5, label=f'Reta Elástica (Ponto Zero = {ponto_zero_pct:.2f}%)')

    # 2. Reta de Offset Móvel (Paralela e vinculada ao Ponto Zero)
    x_off_max = offset_total + (y_lim_max / E_mpa) * 100.0
    x_off = np.linspace(offset_total, x_off_max, 100)
    y_off = E_mpa * ((x_off - offset_total) / 100.0)
    ax.plot(x_off, y_off, 'r--', alpha=0.8, linewidth=3.0, label=f'Reta Offset (+{offset_pct:.2f}%)')

    # Destaque dos Pontos Críticos (Tamanho dos marcadores dobrados)
    ax.scatter(arr_eps[idx_max], sigma_max, color='red', s=250, zorder=5, 
               label=f'Tensão Máxima / UTS ({sigma_max:.1f} MPa)')
    
    if len(cruzamentos) > 0:
        ax.scatter(deformacao_escoamento, tensao_escoamento, color='orange', s=280, zorder=6, 
                   label=f'Escoamento {offset_pct:.2f}% ({tensao_escoamento:.1f} MPa)')
        
        # Linhas de referência dinâmicas (pontilhadas) que acompanham a intersecção
        ax.hlines(y=tensao_escoamento, xmin=0, xmax=deformacao_escoamento, color='orange', linestyle=':', linewidth=2.5, zorder=4)
        ax.vlines(x=deformacao_escoamento, ymin=0, ymax=tensao_escoamento, color='orange', linestyle=':', linewidth=2.5, zorder=4)

    # --- AJUSTE PROPORCIONAL DAS FONTES ---
    ax.set_title('Curva Tensão-Deformação de Engenharia (Aço)', fontsize=26, fontweight='bold', pad=20)
    ax.set_xlabel('Deformação (%)', fontsize=20, labelpad=15)
    ax.set_ylabel('Tensão (N/mm² ou MPa)', fontsize=20, labelpad=15)

    # Ajuste do tamanho dos números nos eixos X e Y
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Ajuste preciso dos limites dos eixos
    ax.set_ylim(0, y_lim_max)
    max_x = max(arr_eps) if len(arr_eps) > 0 else 10.0
    ax.set_xlim(0, max_x * 1.05)

    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Legenda ajustada
    ax.legend(loc='lower right', fontsize=18, frameon=True, facecolor='white', framealpha=0.9)
    fig.tight_layout()

    resultados = {
        "A0": A0, "Af": Af,
        "carga_max_n": carga_max_n,
        "carga_escoamento_n": carga_escoamento_n,
        "sigma_max": sigma_max,
        "tensao_escoamento": tensao_escoamento,
        "razao_elastica": razao_elastica,
        "desloc_max": desloc_max,
        "alongamento_pct": alongamento_pct,
        "estriccao_pct": estriccao_pct,
        "ponto_zero_pct": ponto_zero_pct
    }

    return resultados, fig


def gerar_relatorio_pdf(resultados, w0, t0, L0, wf, tf, Lf, fig, E_gpa, offset_pct):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)

    pdf.cell(0, 10, txt="Relatorio de Ensaio de Tracao em Aco (ASTM E8M)", ln=True, align='C')
    
    # Inserção do autor no PDF
    pdf.set_font("Arial", 'I', 11)
    pdf.cell(0, 6, txt="Autor: Derli da Rosa - Graduado em Automacao.", ln=True, align='C')
    pdf.ln(5)

    # Geometria
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt="1. Geometria do Corpo de Prova", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 6, txt=f"Inicial: Largura (w0) = {w0:.2f} mm | Espessura (t0) = {t0:.2f} mm | L0 = {L0:.2f} mm", ln=True)
    pdf.cell(0, 6, txt=f"Final:   Largura (wf) = {wf:.2f} mm | Espessura (tf) = {tf:.2f} mm | Lf = {Lf:.2f} mm", ln=True)
    pdf.ln(5)

    # Propriedades
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt="2. Propriedades Mecanicas Obtidas", ln=True)
    pdf.set_font("Arial", '', 11)

    propriedades = [
        ("Area Inicial (A0)", f"{resultados['A0']:.2f}", "mm2"),
        ("Modulo de Elasticidade (E)", f"{E_gpa:.0f}", "GPa"),
        ("Correcao Ponto Zero", f"{resultados['ponto_zero_pct']:.2f}", "%"),
        ("Carga de Escoamento (Fe)", f"{resultados['carga_escoamento_n']:.1f}", "N"),
        ("Carga Maxima (Fmax)", f"{resultados['carga_max_n']:.1f}", "N"),
        (f"Tensao de Escoamento ({offset_pct:.2f}%)", f"{resultados['tensao_escoamento']:.1f}", "MPa"),
        ("Tensao Maxima / UTS (Rm)", f"{resultados['sigma_max']:.1f}", "MPa"),
        ("Razao Elastica (Fe/Fmax)", f"{resultados['razao_elastica']:.2f}", "-"),
        ("Alongamento Final (EL)", f"{resultados['alongamento_pct']:.1f}", "%"),
        ("Estriccao / Reducao de Area (RA)", f"{resultados['estriccao_pct']:.1f}", "%")
    ]

    for prop, valor, unid in propriedades:
        pdf.cell(90, 8, txt=prop, border=1)
        pdf.cell(35, 8, txt=valor, border=1, align='C')
        pdf.cell(40, 8, txt=unid, border=1, align='C', ln=True)

    pdf.ln(5)

    # Gráfico
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt="3. Curva Tensao-Deformacao", ln=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
        fig.savefig(tmp_img.name, format="png", bbox_inches="tight", dpi=150)
        pdf.image(tmp_img.name, x=10, w=190)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        with open(tmp_pdf.name, "rb") as f:
            pdf_bytes = f.read()

    return pdf_bytes


# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="Análise de Ensaio de Tração - Aço", layout="wide")
st.title("🔩 Análise Interativa de Ensaio de Tração em Aço (ASTM E8M)")

# Inserção do autor na interface do Streamlit
st.markdown("**Autor:** Derli da Rosa (Graduado em Automação)")
st.markdown("---")

# Sidebar - Dimensões Padrão para Aço
st.sidebar.header("📏 Geometria do Corpo de Prova (Aço)")
w0 = st.sidebar.number_input("Largura Inicial - w0 (mm)", value=12.50, step=0.01)
t0 = st.sidebar.number_input("Espessura Inicial - t0 (mm)", value=3.00, step=0.01)
L0 = st.sidebar.number_input("Comprimento Útil Inicial - L0 (mm)", value=50.00, step=0.10)
st.sidebar.divider()
wf = st.sidebar.number_input("Largura Final - wf (mm)", value=10.20, step=0.01)
tf = st.sidebar.number_input("Espessura Final - tf (mm)", value=2.30, step=0.01)
Lf = st.sidebar.number_input("Comprimento Útil Final - Lf (mm)", value=61.00, step=0.10)

st.sidebar.divider()
st.sidebar.header("🎛️ Ajustes do Aço (Módulo e Escoamento)")

# AQUI ESTÁ A ALTERAÇÃO: O passo (step) agora é 1.0
E_gpa = st.sidebar.slider("Módulo de Elasticidade - E (GPa)", min_value=0.0, max_value=250.0, value=200.0, step=1.0)

# NOVO CONTROLE: Corrige o ponto zero da reta elástica
ponto_zero_pct = st.sidebar.number_input(
    "Correção do Ponto Zero (%)", 
    value=0.00, 
    step=0.01,
    help="Desloca a reta elástica original para frente (compensa a acomodação inicial das garras)."
)

offset_pct = st.sidebar.slider(
    "Offset da Reta (%)", 
    min_value=0.0, 
    max_value=7.0, 
    value=0.2, 
    step=0.01,
    help="Define o deslocamento paralelo em relação ao Ponto Zero. Para aços o padrão é 0,2%."
)

# Entrada de Dados simulados para Aço
st.subheader("Entrada de Dados Brutos (Aço)")
dados_padrao_aco = (
    "0.00   0.000\n"
    "2.50   0.033\n"
    "5.00   0.067\n"
    "7.50   0.100\n"
    "10.00  0.133\n"
    "11.25  0.160\n"
    "12.00  0.250\n"
    "13.50  0.500\n"
    "15.00  1.000\n"
    "17.00  2.000\n"
    "18.00  3.500\n"
    "18.50  5.000\n"
    "17.50  7.000\n"
    "16.00  9.000\n"
    "14.00  9.500"
)

dados_input = st.text_area(
    "Cole as colunas de Carga (kN) e Deslocamento (mm) separadas por espaço ou tabulação:",
    value=dados_padrao_aco, 
    height=200
)

# Processamento dos dados
dados_ensaio = []
for linha in dados_input.strip().split("\n"):
    valores = linha.split()
    if len(valores) == 2:
        try:
            dados_ensaio.append((float(valores[0]), float(valores[1])))
        except ValueError:
            continue

if len(dados_ensaio) > 0:
    resultados, fig = processar_ensaio(dados_ensaio, w0, t0, L0, wf, tf, Lf, E_gpa, offset_pct, ponto_zero_pct)

    st.pyplot(fig, use_container_width=True)
    st.markdown("---")

    col_res1, col_res2 = st.columns(2)

    with col_res1:
        st.subheader("Resultados Gerais")
        st.metric("Área Inicial (A0)", f"{resultados['A0']:.2f} mm²")
        st.metric("Carga Máxima (Fmax)", f"{resultados['carga_max_n']:.1f} N")
        st.metric("Tensão Máxima (UTS)", f"{resultados['sigma_max']:.1f} MPa")
        st.metric("Ponto Zero Ajustado", f"{resultados['ponto_zero_pct']:.2f} %")

    with col_res2:
        st.subheader("Escoamento e Ductilidade")
        st.metric("Carga Escoamento", f"{resultados['carga_escoamento_n']:.1f} N")
        st.metric(f"Tensão Escoamento ({offset_pct:.2f}%)", f"{resultados['tensao_escoamento']:.1f} MPa")
        st.metric("Razão Elástica", f"{resultados['razao_elastica']:.2f}")
        st.metric("Alongamento (EL)", f"{resultados['alongamento_pct']:.1f} %")
        st.metric("Estricção (RA)", f"{resultados['estriccao_pct']:.1f} %")

    st.subheader("📋 Tabela Resumo")
    tabela_resultados = pd.DataFrame({
        "Propriedade Analisada": [
            "Área Inicial (A0)", 
            "Módulo de Elasticidade (E)",
            "Correção Ponto Zero",
            "Carga de Escoamento (Fe)",
            "Carga Máxima (Fmax)", 
            f"Tensão de Escoamento ({offset_pct:.2f}%)", 
            "Tensão Máxima (UTS / Rm)", 
            "Razão Elástica", 
            "Alongamento Final (EL)",
            "Estricção / Redução de Área (RA)"
        ],
        "Valor Obtido": [
            f"{resultados['A0']:.2f}", 
            f"{E_gpa:.0f}",
            f"{resultados['ponto_zero_pct']:.2f}",
            f"{resultados['carga_escoamento_n']:.1f}",
            f"{resultados['carga_max_n']:.1f}", 
            f"{resultados['tensao_escoamento']:.1f}", 
            f"{resultados['sigma_max']:.1f}", 
            f"{resultados['razao_elastica']:.2f}", 
            f"{resultados['alongamento_pct']:.1f}",
            f"{resultados['estriccao_pct']:.1f}"
        ],
        "Unidade de Medida": ["mm²", "GPa", "%", "N", "N", "MPa", "MPa", "-", "%", "%"]
    })

    st.table(tabela_resultados)

    # Relatório PDF
    pdf_bytes = gerar_relatorio_pdf(resultados, w0, t0, L0, wf, tf, Lf, fig, E_gpa, offset_pct)

    st.markdown("---")
    st.download_button(
        label="📄 Baixar Relatório em PDF",
        data=pdf_bytes,
        file_name="Relatorio_Tracao_Aco.pdf",
        mime="application/pdf",
        type="primary"
    )
else:
    st.error("Insira dados válidos de Carga (kN) e Deslocamento (mm) para realizar os cálculos.")

# --- EXECUÇÃO DIRETA NO VS CODE ---
if __name__ == "__main__":
    if os.environ.get("STREAMLIT_RODANDO") != "true":
        os.environ["STREAMLIT_RODANDO"] = "true"
        subprocess.run([sys.executable, "-m", "streamlit", "run", sys.argv[0]])
        sys.exit()