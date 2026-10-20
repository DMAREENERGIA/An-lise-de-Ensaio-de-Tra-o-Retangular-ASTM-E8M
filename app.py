import os
import sys
import io
import tempfile
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from fpdf import FPDF


def processar_ensaio_unico(dados_curva, w0, t0, L0, wf, tf, Lf, E_gpa, offset_pct, ponto_zero_pct):
    """Processa um único ensaio e retorna os dados calculados."""
    if not dados_curva:
        return None

    A0 = w0 * t0
    Af = wf * tf
    E_mpa = E_gpa * 1000.0  

    alongamento_pct = ((Lf - L0) / L0) * 100.0 if L0 > 0 else 0
    estriccao_pct = ((A0 - Af) / A0) * 100.0 if A0 > 0 else 0

    historico_deformacoes = []
    historico_tensoes = []

    for carga_kn, desloc_mm in dados_curva:
        carga_n = carga_kn * 1000.0
        tensao_n_mm2 = carga_n / A0 if A0 > 0 else 0
        deformacao_pct = (desloc_mm / L0) * 100.0 if L0 > 0 else 0

        historico_tensoes.append(tensao_n_mm2)
        historico_deformacoes.append(deformacao_pct)

    arr_eps = np.array(historico_deformacoes)
    arr_sig = np.array(historico_tensoes)

    if len(arr_sig) == 0:
        return None

    idx_max = np.argmax(arr_sig)
    sigma_max = arr_sig[idx_max]
    carga_max_n = sigma_max * A0

    offset_total = ponto_zero_pct + offset_pct
    y_reta_offset = E_mpa * ((arr_eps - offset_total) / 100.0)
    diferenca = arr_sig - y_reta_offset

    cruzamentos = np.where(np.diff(np.sign(diferenca)))[0]

    if len(cruzamentos) > 0:
        i = cruzamentos[0]
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

    return {
        "A0": A0, "Af": Af,
        "carga_max_n": carga_max_n,
        "carga_escoamento_n": carga_escoamento_n,
        "sigma_max": sigma_max,
        "tensao_escoamento": tensao_escoamento,
        "deformacao_escoamento": deformacao_escoamento,
        "razao_elastica": razao_elastica,
        "alongamento_pct": alongamento_pct,
        "estriccao_pct": estriccao_pct,
        "ponto_zero_pct": ponto_zero_pct,
        "arr_eps": arr_eps,
        "arr_sig": arr_sig,
        "idx_max": idx_max
    }


def gerar_grafico_comparativo(resultados_dict, E_gpa, offset_pct, ponto_zero_pct):
    """Gera um gráfico comparativo sobrepondo as curvas dos ensaios."""
    fig, ax = plt.subplots(figsize=(18, 10))
    
    cores = ['#1f77b4', '#ff7f0e', '#2ca02c']
    y_max_global = 0
    x_max_global = 10.0

    for idx, (nome_cp, res) in enumerate(resultados_dict.items()):
        if res is None:
            continue
            
        cor = cores[idx % len(cores)]
        
        ax.plot(res["arr_eps"], res["arr_sig"], color=cor, linestyle='-', 
                linewidth=3.0, label=f'{nome_cp}: Curva Tensão-Deformação')

        ax.scatter(res["arr_eps"][res["idx_max"]], res["sigma_max"], color=cor, 
                   s=180, marker='o', zorder=5, label=f'{nome_cp}: UTS ({res["sigma_max"]:.1f} MPa)')
        
        ax.scatter(res["deformacao_escoamento"], res["tensao_escoamento"], color=cor, 
                   s=180, marker='s', zorder=6, label=f'{nome_cp}: Escoamento ({res["tensao_escoamento"]:.1f} MPa)')

        if res["sigma_max"] > y_max_global:
            y_max_global = res["sigma_max"]
        if len(res["arr_eps"]) > 0 and max(res["arr_eps"]) > x_max_global:
            x_max_global = max(res["arr_eps"])

    E_mpa = E_gpa * 1000.0
    y_lim_max = y_max_global * 1.15 if y_max_global > 0 else 600.0
    offset_total = ponto_zero_pct + offset_pct

    x_lin_max = ponto_zero_pct + (y_lim_max / E_mpa) * 100.0
    x_lin = np.linspace(ponto_zero_pct, x_lin_max, 100)
    y_lin = E_mpa * ((x_lin - ponto_zero_pct) / 100.0)
    ax.plot(x_lin, y_lin, 'k--', alpha=0.5, linewidth=2.0, label=f'Reta Elástica Teórica (E={E_gpa:.0f} GPa)')

    x_off_max = offset_total + (y_lim_max / E_mpa) * 100.0
    x_off = np.linspace(offset_total, x_off_max, 100)
    y_off = E_mpa * ((x_off - offset_total) / 100.0)
    ax.plot(x_off, y_off, 'r--', alpha=0.6, linewidth=2.0, label=f'Reta Offset (+{offset_pct:.2f}%)')

    ax.set_title('Comparação de Ensaios de Tração em Aço (ASTM E8M)', fontsize=20, fontweight='bold', pad=15)
    ax.set_xlabel('Deformação (%)', fontsize=16, labelpad=12)
    ax.set_ylabel('Tensão (N/mm² ou MPa)', fontsize=16, labelpad=12)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_ylim(0, y_lim_max)
    ax.set_xlim(0, x_max_global * 1.05)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    ax.legend(loc='lower right', fontsize=11, frameon=True, facecolor='white', framealpha=0.9)
    fig.tight_layout()

    return fig


def gerar_grafico_uts_barras(resultados_dict, offset_pct):
    """Gera gráfico de barras do UTS de cada CP e a Média com Barra de Erro em 300 DPI."""
    nomes_cp = list(resultados_dict.keys())
    valores_uts = [res["sigma_max"] for res in resultados_dict.values()]

    if len(valores_uts) == 0:
        return None, None

    media_uts = np.mean(valores_uts)
    desvio_uts = np.std(valores_uts, ddof=1) if len(valores_uts) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    cores_cps = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(nomes_cp)]

    bars = ax.bar(nomes_cp, valores_uts, color=cores_cps, alpha=0.85, width=0.45, edgecolor='black', linewidth=1.2)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + (yval * 0.01), f"{yval:.1f} MPa", 
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    if len(valores_uts) > 1:
        bar_media = ax.bar(['Média Geral'], [media_uts], color='#7f7f7f', alpha=0.9, width=0.45, 
                           edgecolor='black', linewidth=1.2,
                           yerr=[desvio_uts], capsize=8, ecolor='red', error_kw={'linewidth': 2})
        
        ax.text(bar_media[0].get_x() + bar_media[0].get_width()/2.0, media_uts + desvio_uts + (media_uts * 0.01), 
                f"{media_uts:.1f} ± {desvio_uts:.1f} MPa", 
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='darkred')

    ax.set_title('Comparativo de Tensão Máxima / UTS (MPa)', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('Tensão Máxima - UTS (MPa)', fontsize=12)
    ax.tick_params(axis='both', labelsize=11)
    
    y_max = max(valores_uts) + (desvio_uts if len(valores_uts) > 1 else 0)
    ax.set_ylim(0, y_max * 1.22)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    fig.tight_layout()

    buffer_img = io.BytesIO()
    fig.savefig(buffer_img, format='png', dpi=300, bbox_inches='tight')
    buffer_img.seek(0)

    return fig, buffer_img.getvalue()


def gerar_relatorio_pdf_comparativo(df_comparativo, fig_comp, fig_uts, E_gpa, offset_pct, ponto_zero_pct):
    """Gera o relatório técnico final em PDF com tabelas e figuras."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Cabeçalho do Documento
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="Relatorio Tecnico Final: Ensaio de Tracao em Aco", ln=True, align='C')
    
    pdf.set_font("Arial", 'I', 11)
    pdf.cell(0, 6, txt="Norma de Referencia: ASTM E8M | Autor: Derli da Rosa", ln=True, align='C')
    pdf.ln(5)

    # Seção 1: Parâmetros do Ensaio
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt="1. Parametros Globais Configurados", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 5, txt=f"Modulo de Elasticidade (E): {E_gpa:.0f} GPa | Offset: {offset_pct:.2f}% | Correcao Ponto Zero: {ponto_zero_pct:.2f}%", ln=True)
    pdf.ln(4)

    # Seção 2: Tabela Comparativa de Resultados
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt="2. Tabela Resumo Comparativa de Propriedades Mecanicas", ln=True)
    
    # Formatação da Tabela no PDF
    pdf.set_font("Arial", 'B', 8)
    
    num_cols = len(df_comparativo.columns)
    col_widths = [45, 18] + [int(125 / (num_cols - 2))] * (num_cols - 2)

    # Cabeçalho da Tabela
    for col_idx, col_name in enumerate(df_comparativo.columns):
        nome_limpo = str(col_name).replace("Á", "A").replace("ã", "a").replace("ó", "o").replace("ç", "c").replace("±", "+/-")
        pdf.cell(col_widths[col_idx], 7, txt=nome_limpo, border=1, align='C')
    pdf.ln()

    # Linhas da Tabela
    pdf.set_font("Arial", '', 8)
    for _, row in df_comparativo.iterrows():
        for col_idx, value in enumerate(row):
            val_str = str(value).replace("Á", "A").replace("ã", "a").replace("ó", "o").replace("ç", "c").replace("±", "+/-")
            align_txt = 'L' if col_idx == 0 else 'C'
            pdf.cell(col_widths[col_idx], 6, txt=val_str, border=1, align=align_txt)
        pdf.ln()

    pdf.ln(6)

    # Seção 3: Inserção dos Gráficos
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt="3. Curvas Tensao-Deformacao Comparativas", ln=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_comp:
        fig_comp.savefig(tmp_comp.name, format="png", bbox_inches="tight", dpi=150)
        pdf.image(tmp_comp.name, x=10, w=190)

    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt="4. Comparativo da Tensao Maxima (UTS) com Desvio Padrao", ln=True)

    if fig_uts is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_uts:
            fig_uts.savefig(tmp_uts.name, format="png", bbox_inches="tight", dpi=150)
            pdf.image(tmp_uts.name, x=25, w=160)

    # Converte o PDF para bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        with open(tmp_pdf.name, "rb") as f:
            pdf_bytes = f.read()

    return pdf_bytes


def parse_texto_dados(texto):
    """Trata dados em formato de texto aceitando vírgula ou ponto decimal."""
    dados = []
    for linha in texto.strip().split("\n"):
        valores = linha.split()
        if len(valores) >= 2:
            try:
                carga = float(valores[0].replace(',', '.'))
                desloc = float(valores[1].replace(',', '.'))
                dados.append((carga, desloc))
            except ValueError:
                continue
    return dados


# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="Análise Comparativa de Tração - Aço", layout="wide")
st.title("🔩 Análise Comparativa de Ensaios de Tração em Aço (ASTM E8M)")
st.markdown("**Autor:** Derli da Rosa (Graduado em Automação)")
st.markdown("---")

num_ensaios = st.radio("Selecione o número de corpos de prova para comparar:", [1, 2, 3], horizontal=True)

geometria = {}
dados_cp = {}

for i in range(1, num_ensaios + 1):
    st.sidebar.header(f"📏 Geometria do Corpo de Prova {i}")
    w0 = st.sidebar.number_input(f"CP {i} - Largura Inicial w0 (mm)", value=12.50, step=0.01, key=f"w0_{i}")
    t0 = st.sidebar.number_input(f"CP {i} - Espessura Inicial t0 (mm)", value=3.00, step=0.01, key=f"t0_{i}")
    L0 = st.sidebar.number_input(f"CP {i} - Comprimento Útil L0 (mm)", value=50.00, step=0.10, key=f"L0_{i}")
    wf = st.sidebar.number_input(f"CP {i} - Largura Final wf (mm)", value=10.20, step=0.01, key=f"wf_{i}")
    tf = st.sidebar.number_input(f"CP {i} - Espessura Final tf (mm)", value=2.30, step=0.01, key=f"tf_{i}")
    Lf = st.sidebar.number_input(f"CP {i} - Comprimento Útil Final Lf (mm)", value=61.00, step=0.10, key=f"Lf_{i}")
    st.sidebar.divider()

    geometria[f"CP {i}"] = (w0, t0, L0, wf, tf, Lf)

st.sidebar.header("🎛️ Parâmetros Materiais Globais")
E_gpa = st.sidebar.slider("Módulo de Elasticidade - E (GPa)", min_value=0.0, max_value=250.0, value=200.0, step=1.0)
ponto_zero_pct = st.sidebar.number_input("Correção do Ponto Zero (%)", value=0.00, step=0.01)
offset_pct = st.sidebar.slider("Offset da Reta (%)", min_value=0.0, max_value=7.0, value=0.2, step=0.01)

st.subheader("Entrada de Dados Brutos (Carga em kN / Deslocamento em mm)")
abas_cps = st.tabs([f"Corpo de Prova {i}" for i in range(1, num_ensaios + 1)])

dados_padrao = (
    "0.00   0.000\n2.50   0.033\n5.00   0.067\n7.50   0.100\n10.00  0.133\n"
    "11.25  0.160\n12.00  0.250\n13.50  0.500\n15.00  1.000\n17.00  2.000\n"
    "18.00  3.500\n18.50  5.000\n17.50  7.000\n16.00  9.000\n14.00  9.500"
)

for i, aba in enumerate(abas_cps, start=1):
    nome_cp = f"CP {i}"
    with aba:
        sub_tab_txt, sub_tab_arq = st.tabs(["📋 Colar Texto", "📁 Upload Arquivo"])
        with sub_tab_txt:
            txt_in = st.text_area(f"Cole os dados do {nome_cp}:", value=dados_padrao if i == 1 else "", height=150, key=f"txt_{i}")
            if txt_in.strip():
                dados_cp[nome_cp] = parse_texto_dados(txt_in)
        with sub_tab_arq:
            arq_in = st.file_uploader(f"Upload CSV/Excel para {nome_cp}", type=["csv", "xlsx", "xls"], key=f"arq_{i}")
            if arq_in is not None:
                try:
                    if arq_in.name.endswith('.csv'):
                        df = pd.read_csv(arq_in, sep=None, engine='python', decimal=',')
                    else:
                        df = pd.read_excel(arq_in)
                    df = df.iloc[:, :2].dropna()
                    lista_dados = []
                    for _, row in df.iterrows():
                        c = float(str(row.iloc[0]).replace(',', '.'))
                        d = float(str(row.iloc[1]).replace(',', '.'))
                        lista_dados.append((c, d))
                    dados_cp[nome_cp] = lista_dados
                    st.success(f"{nome_cp}: {len(lista_dados)} linhas carregadas.")
                except Exception as e:
                    st.error(f"Erro ao ler arquivo do {nome_cp}: {e}")

resultados = {}
for nome_cp, dados in dados_cp.items():
    if dados:
        w0, t0, L0, wf, tf, Lf = geometria[nome_cp]
        res = processar_ensaio_unico(dados, w0, t0, L0, wf, tf, Lf, E_gpa, offset_pct, ponto_zero_pct)
        if res:
            resultados[nome_cp] = res

if resultados:
    aba_curvas, aba_barras = st.tabs(["📈 Curvas Tensão-Deformação", "📊 Comparativo UTS (Barras de Erro)"])

    with aba_curvas:
        fig_comp = gerar_grafico_comparativo(resultados, E_gpa, offset_pct, ponto_zero_pct)
        st.pyplot(fig_comp, use_container_width=True)

    with aba_barras:
        fig_uts, img_png_bytes = gerar_grafico_uts_barras(resultados, offset_pct)
        if fig_uts:
            st.pyplot(fig_uts, use_container_width=True)
            st.download_button(
                label="🖼️ Baixar Gráfico UTS em HD (PNG 300 DPI)",
                data=img_png_bytes,
                file_name="Grafico_UTS_300DPI.png",
                mime="image/png",
                type="secondary"
            )

    st.markdown("---")

    st.subheader("📋 Tabela Comparativa de Propriedades Mecânicas")
    
    propriedades_chaves = [
        ("Área Inicial (A0)", "mm²", "A0", 2),
        ("Carga de Escoamento (Fe)", "N", "carga_escoamento_n", 1),
        ("Carga Máxima (Fmax)", "N", "carga_max_n", 1),
        (f"Tensão de Escoamento ({offset_pct:.2f}%)", "MPa", "tensao_escoamento", 1),
        ("Tensão Máxima (UTS)", "MPa", "sigma_max", 1),
        ("Razão Elástica", "-", "razao_elastica", 2),
        ("Alongamento Final (EL)", "%", "alongamento_pct", 1),
        ("Estricção (RA)", "%", "estriccao_pct", 1)
    ]

    tabela_dict = {
        "Propriedade Analisada": [p[0] for p in propriedades_chaves], 
        "Unidade": [p[1] for p in propriedades_chaves]
    }

    matriz_valores = []

    for nome_cp, res in resultados.items():
        valores_num = []
        valores_formatados = []
        
        for prop_nome, unid, chave, decimais in propriedades_chaves:
            val = res[chave]
            valores_num.append(val)
            valores_formatados.append(f"{val:.{decimais}f}")
            
        tabela_dict[nome_cp] = valores_formatados
        matriz_valores.append(valores_num)

    matriz_valores = np.array(matriz_valores)

    col_media = []
    col_desvio = []

    for idx, (prop_nome, unid, chave, decimais) in enumerate(propriedades_chaves):
        vetor_propriedade = matriz_valores[:, idx]
        
        media = np.mean(vetor_propriedade)
        col_media.append(f"{media:.{decimais}f}")
        
        if len(vetor_propriedade) > 1:
            desvio = np.std(vetor_propriedade, ddof=1)
            col_desvio.append(f"{desvio:.{decimais}f}")
        else:
            col_desvio.append("0.00")

    tabela_dict["Média"] = col_media
    tabela_dict["Desvio Padrão (±)"] = col_desvio

    df_comparativo = pd.DataFrame(tabela_dict)
    st.table(df_comparativo)

    st.markdown("---")
    st.subheader("📥 Exportação de Relatórios")

    col_pdf, col_excel = st.columns(2)

    with col_pdf:
        # --- EXPORTAÇÃO PDF RELATÓRIO FINAL ---
        pdf_bytes = gerar_relatorio_pdf_comparativo(df_comparativo, fig_comp, fig_uts, E_gpa, offset_pct, ponto_zero_pct)
        st.download_button(
            label="📄 Baixar Relatório Final Completo (PDF)",
            data=pdf_bytes,
            file_name="Relatorio_Final_Ensaio_Tracao.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

    with col_excel:
        # --- EXPORTAÇÃO EXCEL COMPARATIVO ---
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_comparativo.to_excel(writer, index=False, sheet_name='Resumo_Comparativo')
            
            dict_curvas = {}
            for nome_cp, res in resultados.items():
                dict_curvas[f"{nome_cp} - Def (%)"] = res["arr_eps"]
                dict_curvas[f"{nome_cp} - Ten (MPa)"] = res["arr_sig"]
            
            max_len = max(len(v) for v in dict_curvas.values())
            for k, v in dict_curvas.items():
                if len(v) < max_len:
                    pad = np.full(max_len - len(v), np.nan)
                    dict_curvas[k] = np.concatenate([v, pad])

            df_curvas = pd.DataFrame(dict_curvas)
            df_curvas.to_excel(writer, index=False, sheet_name='Curvas_Calculadas')

        st.download_button(
            label="📊 Baixar Comparativo Completo em Excel (.xlsx)",
            data=buffer.getvalue(),
            file_name="Comparativo_Ensaios_Tracao.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="secondary",
            use_container_width=True
        )

else:
    st.info("Insira os dados de pelo menos 1 corpo de prova para visualizar o gráfico e os resultados.")

# --- EXECUÇÃO DIRETA NO VS CODE ---
if __name__ == "__main__":
    if os.environ.get("STREAMLIT_RODANDO") != "true":
        os.environ["STREAMLIT_RODANDO"] = "true"
        subprocess.run([sys.executable, "-m", "streamlit", "run", sys.argv[0]])
        sys.exit()