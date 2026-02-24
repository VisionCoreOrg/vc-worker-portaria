import json
import os

def validar_tamanho_placas(arquivo_json_entrada, arquivo_json_saida):
    if not os.path.exists(arquivo_json_entrada):
        print(f"Erro: O arquivo '{arquivo_json_entrada}' não foi encontrado.")
        return

    print(f"Carregando dados de '{arquivo_json_entrada}'...")
    
    # Lê os dados do arquivo JSON original
    with open(arquivo_json_entrada, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    total_processado = len(dados)
    
    if total_processado == 0:
        print("O arquivo JSON está vazio.")
        return

    # Contadores para as estatísticas
    total_validas = 0
    erros_maior_7 = 0
    erros_menor_7 = 0

    print("-" * 50)
    print("Processando validações de tamanho...\n")
    
    # Itera sobre cada objeto no JSON
    for item in dados:
        texto_placa = item.get("placa", "")
        tamanho = len(texto_placa)
        
        # Regra 1: Maior que 7 (OCR leu lixo junto com a placa)
        if tamanho > 7:
            item["erro"] = True
            item["motivo_erro"] = "Comprimento maior que 7"
            print(f"[ERRO > 7] ID {item['id']}: '{texto_placa}' ({tamanho} caracteres)")
            erros_maior_7 += 1
            
        # Regra 2: Menor que 7 (OCR não conseguiu ler a placa inteira)
        elif tamanho < 7:
            item["erro"] = True
            item["motivo_erro"] = "Comprimento menor que 7"
            print(f"[ERRO < 7] ID {item['id']}: '{texto_placa}' ({tamanho} caracteres)")
            erros_menor_7 += 1
            
        # Regra 3: Exatamente 7 (Tamanho ideal do padrão Brasil/Mercosul)
        else:
            item["erro"] = False
            item["motivo_erro"] = None
            total_validas += 1

    # Cálculos Estatísticos
    total_erros = erros_maior_7 + erros_menor_7
    
    taxa_acuracia = (total_validas / total_processado) * 100
    taxa_erro_geral = (total_erros / total_processado) * 100
    taxa_erro_maior = (erros_maior_7 / total_processado) * 100
    taxa_erro_menor = (erros_menor_7 / total_processado) * 100

    # Exibição do Dashboard no Terminal
    print("\n" + "=" * 50)
    print("📊 RELATÓRIO ESTATÍSTICO DE EXTRAÇÃO (OCR)")
    print("=" * 50)
    print(f"Total de placas processadas: {total_processado}")
    print(f"Placas perfeitas (7 dígitos):  {total_validas}  [{taxa_acuracia:.1f}%]")
    print("-" * 50)
    print(f"Taxa de Erro Total:            {total_erros}  [{taxa_erro_geral:.1f}%]")
    print(f"  ↳ Erros por excesso (> 7):   {erros_maior_7}   [{taxa_erro_maior:.1f}%]")
    print(f"  ↳ Erros por omissão (< 7):   {erros_menor_7}   [{taxa_erro_menor:.1f}%]")
    print("=" * 50)

    # Salva os dados atualizados em um novo arquivo JSON
    with open(arquivo_json_saida, 'w', encoding='utf-8') as f:
        json.dump(dados, f, ensure_ascii=False, indent=4)
        
    print(f"\n📁 Arquivo validado salvo com sucesso em '{arquivo_json_saida}'.")

if __name__ == "__main__":
    # O arquivo gerado no passo anterior
    ARQUIVO_ENTRADA = "dados_placas.json" 
    
    # O novo arquivo que será gerado com as marcações de erro
    ARQUIVO_SAIDA = "dados_placas_validados.json"
    
    validar_tamanho_placas(ARQUIVO_ENTRADA, ARQUIVO_SAIDA)