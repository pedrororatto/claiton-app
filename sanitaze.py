import re
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import PyPDF2


@dataclass
class Acordao:
    """Estrutura de dados para um acórdão processado"""
    tribunal: Optional[str] = None
    orgao_julgador: Optional[str] = None
    processo: Optional[str] = None
    relator: Optional[str] = None
    data: Optional[str] = None
    crime: Optional[str] = None
    ementa: Optional[str] = None
    fundamentos: List[str] = None
    decisao: Optional[str] = None
    texto_integral: Optional[str] = None
    fonte: Optional[str] = None
    caminho_arquivo: Optional[str] = None

    def __post_init__(self):
        if self.fundamentos is None:
            self.fundamentos = []


class SanitizadorJurisprudencia:
    """Classe principal para sanitização de documentos jurídicos"""

    # Mapeamento de crimes para detecção automática
    CRIMES_KEYWORDS = {
        'Estelionato': ['estelionato', 'fraude', 'engano'],
        'Tráfico': ['tráfico', 'narcotráfico', 'entorpecente', 'droga', 'lei 11.343', 'lei de drogas'],
        'Furto': ['furto', 'subtração'],
        'Lesão Corporal': ['lesão corporal', 'lesões corporais', 'agressão física'],
        'Porte/Consumo': ['porte', 'consumo', 'uso de droga', 'usuário'],
        'Embriaguez': ['embriaguez', 'embriagado', 'alcoolizado', 'direção sob efeito', 'art. 306 do ctb'],
        'Homicídio': ['homicídio', 'homicidio', 'morte', 'latrocínio'],
        'Roubo': ['roubo', 'assalto'],
        'Receptação': ['receptação', 'receptacao', 'produto de crime']
    }

    # Padrões de ruído a serem removidos (linha por linha)
    NOISE_PATTERNS = [
        r'^\s*Baixado do vLex.*$',
        r'^\s*©\s*Copyright.*$',
        r'^\s*C[oó]pia exclusiva.*$',
        r'^\s*vLex Document Id:.*$',
        r'^\s*Link:\s*https?://\S+.*$',
        r'^\s*\d{1,2}\s+de\s+[A-Za-zçãõéíóú]+\s+de\s+\d{4}\s+\d{2}:\d{2}\s*$',
        r'^\s*\d+\s*/\s*\d+\s*$',
        r'^\s*vLex\s*$',
        r'^\s*Resumo\s*$',
        r'^\s*[-_\.]{3,}\s*$',
        r'^\s*P[aá]gina\s*\d+\s*de\s*\d+\s*$',
    ]

    def __init__(self):
        self.stats = {
            'processados': 0,
            'erros': 0,
            'chunks_gerados': 0
        }

    def extrair_texto_pdf(self, caminho_pdf: str) -> str:
        """Extrai texto bruto de um PDF"""
        try:
            with open(caminho_pdf, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                texto = []
                for page in reader.pages:
                    try:
                        t = page.extract_text() or ""
                    except Exception:
                        t = ""
                    texto.append(t)
                return "\n".join(texto)
        except Exception as e:
            print(f"[ERRO] Falha ao ler PDF {caminho_pdf}: {e}")
            return ""

    def sanitizar_texto(self, texto: str) -> str:
        """Remove ruídos e normaliza o texto"""
        # Remoção de ruídos por linha
        linhas = texto.splitlines()
        linhas_limpas = []

        for l in linhas:
            descartar = False
            # Verifica se a linha corresponde a algum padrão de ruído
            for pattern in self.NOISE_PATTERNS:
                if re.search(pattern, l, flags=re.IGNORECASE):
                    descartar = True
                    break

            if descartar:
                continue

            # Remover linhas muito curtas que são lixo (mas preserve parágrafos vazios)
            if l.strip() and len(l.strip()) <= 2:
                continue

            linhas_limpas.append(l)

        texto = "\n".join(linhas_limpas)

        # Normalizações
        texto = re.sub(r'\u00a0', ' ', texto)  # non-breaking space
        texto = re.sub(r'[ \t]+', ' ', texto)  # múltiplos espaços → um espaço
        texto = re.sub(r'\n[ \t]+', '\n', texto)  # remove espaços após quebra
        texto = re.sub(r'\n{3,}', '\n\n', texto)  # múltiplas quebras → duas quebras

        return texto.strip()

    def extrair_metadados(self, texto: str) -> Dict:
        """Extrai metadados estruturados do texto"""
        metadados = {}

        # Fonte (URL vLex) se existir em alguma linha que sobrou
        fonte_match = re.search(r'https?://\S*vlex\.com\S*', texto, re.IGNORECASE)
        if fonte_match:
            metadados['fonte'] = fonte_match.group(0)

        # Tribunal
        if re.search(r'\bSuperior Tribunal de Justi[cç]a\b', texto, re.IGNORECASE) or re.search(r'\bSTJ\b', texto):
            metadados['tribunal'] = 'STJ'
        elif re.search(r'\bSupremo Tribunal Federal\b', texto, re.IGNORECASE) or re.search(r'\bSTF\b', texto):
            metadados['tribunal'] = 'STF'
        else:
            t = re.search(r'\b(TJ[A-Z]{2}|TRF\d)\b', texto)
            if t:
                metadados['tribunal'] = t.group(1)

        # Órgão julgador
        orgao = re.search(r'\b(Primeira|Segunda|Terceira|Quarta|Quinta|Sexta)\s+Turma\b', texto, re.IGNORECASE)
        if orgao:
            metadados['orgao_julgador'] = orgao.group(0).title()
        else:
            sec = re.search(r'\b(Terceira Se[cç][aã]o|Plen[aá]rio)\b', texto, re.IGNORECASE)
            if sec:
                metadados['orgao_julgador'] = sec.group(1).title()

        # Processo (bem permissivo; evita capturar blocos gigantes)
        proc = re.search(r'\b(?:Processo|REsp|HC|RHC|AREsp)\s*[:\-]?\s*([A-Z0-9\.\-\/]+)\b', texto, re.IGNORECASE)
        if proc:
            metadados['processo'] = proc.group(1)
        else:
            vlex = re.search(r'\bVLEX-(\d+)\b', texto, re.IGNORECASE)
            if vlex:
                metadados['processo'] = vlex.group(1)

        # Data "Data: 05 Dezembro 2017" (com ou sem "de" no meio)
        data = re.search(r'Data\s*:\s*([0-3]?\d\s+[A-Za-zçãõéíóú]+(?:\s+de)?\s+\d{4})', texto, re.IGNORECASE)
        if data:
            metadados['data'] = data.group(1)

        # Relator (não muito ganancioso)
        rel = re.search(
            r'(?:Relator(?:a)?|Ministro Relator|Desembargador(?:a)? Relator(?:a)?)\s*[:\-]?\s*([A-ZÁÉÍÓÚÂÊÔÃÕÇ][\w\s\.\-ÁÉÍÓÚÂÊÔÃÕÇ]+?)(?:\n|$)',
            texto,
            re.IGNORECASE
        )
        if rel:
            metadados['relator'] = rel.group(1).strip()

        # Crime
        crime = self.detectar_crime(texto)
        if crime:
            metadados['crime'] = crime

        return metadados

    def detectar_crime(self, texto: str) -> Optional[str]:
        """Detecta o tipo de crime mencionado no texto"""
        t = texto.lower()
        scores = {}

        for crime, kws in self.CRIMES_KEYWORDS.items():
            score = 0
            for kw in kws:
                # Conta ocorrências com word boundary para evitar falsos positivos
                score += len(re.findall(r'\b' + re.escape(kw) + r'\b', t))
            if score > 0:
                scores[crime] = score

        return max(scores, key=scores.get) if scores else None

    def extrair_ementa(self, texto: str) -> Optional[str]:
        """Extrai a ementa do acórdão"""
        try:
            # Aceita "Ementa", "EMENTA", "Resumo", com ou sem dois-pontos, e espaço variável
            padrao = re.compile(
                r'(?:Ementa|EMENTA|Resumo)\s*:?\s*(.*?)(?=\n{2,}|Acord[aã]o|ACORD[AÃ]O|Vistos|Relat[óo]rio|RELAT[ÓO]RIO|Decidem|ACORDAM|Acordam)',
                flags=re.IGNORECASE | re.DOTALL
            )
            m = padrao.search(texto)
            if not m:
                return None

            e = re.sub(r'\s+', ' ', m.group(1)).strip()
            return e if len(e) > 20 else None
        except Exception as ex:
            print(f"[AVISO] Erro ao extrair ementa: {ex}")
            return None

    def extrair_decisao(self, texto: str) -> Optional[str]:
        """Extrai a parte dispositiva da decisão"""
        try:
            padrao = re.compile(
                r'(?:Acord[aã]o|ACORD[AÃ]O|Acordam|ACORDAM|Decidem)\s*:?\s*(.*?)(?=\n{2,}|$)',
                flags=re.IGNORECASE | re.DOTALL
            )
            m = padrao.search(texto)
            if not m:
                return None

            d = re.sub(r'\s+', ' ', m.group(1)).strip()
            return d if len(d) > 10 else None
        except Exception as ex:
            print(f"[AVISO] Erro ao extrair decisão: {ex}")
            return None

    def extrair_fundamentos(self, texto: str) -> List[str]:
        """Extrai fundamentos jurídicos principais"""
        fundamentos = []

        try:
            # Linhas iniciadas por hífen ou travessão com frase significativa
            for m in re.findall(r'[\-\–]\s+([^.\n]{30,400}\.)', texto):
                s = m.strip()
                if 20 < len(s) < 400:
                    fundamentos.append(s)

            # Remoção de duplicados mantendo ordem
            seen = set()
            uniq = []
            for f in fundamentos:
                if f not in seen:
                    uniq.append(f)
                    seen.add(f)

            return uniq[:10]  # Limita a 10 fundamentos principais
        except Exception as ex:
            print(f"[AVISO] Erro ao extrair fundamentos: {ex}")
            return []

    def processar_acordao(self, pdf_path: Path) -> Optional[Acordao]:
        """Processa um PDF de acórdão e retorna objeto estruturado"""
        texto_bruto = self.extrair_texto_pdf(str(pdf_path))
        if not texto_bruto:
            self.stats['erros'] += 1
            return None

        texto_limpo = self.sanitizar_texto(texto_bruto)
        metadados = self.extrair_metadados(texto_limpo)

        acordao = Acordao(
            tribunal=metadados.get('tribunal'),
            orgao_julgador=metadados.get('orgao_julgador'),
            processo=metadados.get('processo'),
            relator=metadados.get('relator'),
            data=metadados.get('data'),
            crime=metadados.get('crime'),
            ementa=self.extrair_ementa(texto_limpo),
            fundamentos=self.extrair_fundamentos(texto_limpo),
            decisao=self.extrair_decisao(texto_limpo),
            texto_integral=texto_limpo,
            fonte=metadados.get('fonte'),
            caminho_arquivo=str(pdf_path)
        )

        self.stats['processados'] += 1
        return acordao

    def gerar_chunks(self, acordao: Acordao, max_tokens: int = 1000) -> List[Dict]:
        """Divide o texto integral em chunks para indexação vetorial"""
        if not acordao.texto_integral:
            return []

        texto = acordao.texto_integral
        max_palavras = int(max_tokens * 0.75)  # ~0,75 palavra/token em português
        palavras = texto.split()
        chunks = []

        for i in range(0, len(palavras), max_palavras):
            parte = ' '.join(palavras[i:i + max_palavras])
            chunks.append({
                'chunk_id': f"{(acordao.processo or Path(acordao.caminho_arquivo).stem)}-{len(chunks) + 1}",
                'crime': acordao.crime,
                'tribunal': acordao.tribunal,
                'orgao_julgador': acordao.orgao_julgador,
                'data': acordao.data,
                'ementa': acordao.ementa,
                'texto': parte,
                'fonte': acordao.fonte,
                'arquivo_origem': acordao.caminho_arquivo
            })
            self.stats['chunks_gerados'] += 1

        return chunks

    def processar_diretorio(self, dir_entrada: str, dir_saida: str, gerar_chunks: bool = True):
        """Processa todos os PDFs de um diretório"""
        entrada = Path(dir_entrada)
        saida = Path(dir_saida)
        saida.mkdir(parents=True, exist_ok=True)

        dir_acordaos = saida / 'acordaos'
        dir_chunks = saida / 'chunks'
        dir_acordaos.mkdir(exist_ok=True)
        if gerar_chunks:
            dir_chunks.mkdir(exist_ok=True)

        pdfs = list(entrada.rglob('*.pdf'))
        print(f"Encontrados {len(pdfs)} PDFs em {entrada}")
        print("=" * 60)

        for idx, pdf in enumerate(pdfs, 1):
            try:
                print(f"[{idx}/{len(pdfs)}] Processando: {pdf.name}")
                acordao = self.processar_acordao(pdf)

                if not acordao:
                    print(f"  ⚠️  Falha ao processar (texto vazio ou erro)")
                    continue

                # Salva acórdão completo
                nome_base = pdf.stem
                with open(dir_acordaos / f"{nome_base}.json", 'w', encoding='utf-8') as f:
                    json.dump(asdict(acordao), f, ensure_ascii=False, indent=2)

                # Gera e salva chunks
                if gerar_chunks:
                    chunks = self.gerar_chunks(acordao)
                    with open(dir_chunks / f"{nome_base}_chunks.json", 'w', encoding='utf-8') as f:
                        json.dump(chunks, f, ensure_ascii=False, indent=2)
                    print(f"  ✓ Gerados {len(chunks)} chunks")
                else:
                    print(f"  ✓ Processado com sucesso")

            except Exception as e:
                print(f"  ❌ ERRO ao processar {pdf.name}: {e}")
                self.stats['erros'] += 1
                continue

        # Relatório final
        print("\n" + "=" * 60)
        print("RELATÓRIO DE PROCESSAMENTO")
        print("=" * 60)
        print(f"✓ PDFs processados com sucesso: {self.stats['processados']}")
        print(f"✗ Erros: {self.stats['erros']}")
        print(f"📦 Chunks gerados: {self.stats['chunks_gerados']}")
        print(f"📁 Arquivos salvos em: {saida}")
        print("=" * 60)

    def validar_json(self, caminho_json: str) -> bool:
        """Valida se um JSON está bem formatado"""
        try:
            with open(caminho_json, 'r', encoding='utf-8') as f:
                json.load(f)
            return True
        except json.JSONDecodeError as e:
            print(f"JSON inválido em {caminho_json}: {e}")
            return False

def main():
    DIRETORIO_PDFS = '/home/roratto/Documents/vlex'
    DIRETORIO_SAIDA = './dados_sanitizados'

    sanitizador = SanitizadorJurisprudencia()
    sanitizador.processar_diretorio(
        dir_entrada=DIRETORIO_PDFS,
        dir_saida=DIRETORIO_SAIDA,
        gerar_chunks=True
    )

if __name__ == '__main__':
    main()