# Otimização de Hiperparâmetros de MLPs usando Algoritmos Genéticos

## 📋 Sobre o Projeto

Este trabalho investiga a aplicação de **Algoritmos Genéticos (GA)** para otimização de hiperparâmetros em **Multi-Layer Perceptrons**, comparando diferentes estratégias evolutivas com uma abordagem baseline tradicional.

### 🎯 Objetivos Principais

- Investigar a eficácia dos Algoritmos Genéticos na otimização de hiperparâmetros de MLPs
- Comparar diferentes configurações de GA com uma abordagem baseline
- Avaliar o trade-off entre melhoria de performance e custo computacional
- Fornecer diretrizes práticas para aplicação de GAs em problemas similares

## 🔬 Metodologia

### Dataset
- **Tipo**: Sintético de classificação binária
- **Amostras**: 1.000 instâncias
- **Features**: 20 atributos (10 informativos, 5 redundantes)
- **Divisão**: 60% treino, 20% validação, 20% teste

### Hiperparâmetros Otimizados

| Parâmetro | Valores Possíveis | Descrição |
|-----------|-------------------|-----------|
| `learning_rate_init` | [0.0001, 0.001, 0.01, 0.1] | Taxa de aprendizado inicial |
| `hidden_layer_sizes` | [(50,), (100,), (50,50), (100,50)] | Arquitetura das camadas ocultas |
| `activation` | ['logistic', 'tanh', 'relu'] | Função de ativação |
| `solver` | ['adam', 'sgd'] | Algoritmo de otimização |
| `max_iter` | [50, 500] | Número máximo de épocas |

### Configurações Experimentais

| Experimento | População | Gerações | Crossover | Mutação | Descrição |
|-------------|-----------|----------|-----------|---------|-----------|
| **Baseline** | - | - | - | - | Parâmetros padrão MLPClassifier |
| **GA_Config1** | 30 | 25 | 0.7 | 0.2 | População pequena, convergência rápida |
| **GA_Config2** | 50 | 50 | 0.7 | 0.2 | Configuração equilibrada |
| **GA_Config3** | 70 | 40 | 0.8 | 0.15 | População grande, alta cooperação |
| **GA_Config4** | 40 | 60 | 0.6 | 0.3 | Alta exploração, mais gerações |

## 📊 Principais Resultados

### Performance
- ✅ **Melhoria consistente**: Todos os GAs superaram o baseline
- ✅ **Melhor configuração**: GA_Config3 com +5.6% no F1-score
- ✅ **Melhoria média**: +4.2% no F1-score sobre baseline

### Custo Computacional
- ⏱️ **Tempo baseline**: ~2.3 segundos
- ⏱️ **Tempo GAs**: 15-68 segundos (7-30x mais lento)
- 💻 **Trade-off**: Melhoria de performance vs custo computacional

### Hiperparâmetros Descobertos
Os GAs encontraram configurações não óbvias:
- Preferência por arquiteturas de duas camadas
- Learning rates mais altos (0.01) em muitos casos
- Combinações específicas de solver + activation

## 🚀 Como Usar

### Pré-requisitos

```bash
pip install numpy pandas matplotlib scikit-learn deap
```

### Execução Rápida

```python
# Clone o repositório
git clone https://github.com/seu-usuario/ga-hyperparameter-optimization.git
cd ga-hyperparameter-optimization

# Execute o notebook
jupyter notebook evolutionary_algorithms.ipynb
```

### Estrutura do Projeto

```
├── evolutionary_algorithms.ipynb    # Notebook principal
├── README.md                              # Este arquivo  
```

### Exemplo de Uso

```python
# Configuração básica de um experimento GA
experimento = {
    'pop_size': 50,      # Tamanho da população
    'ngen': 40,          # Número de gerações
    'cxpb': 0.7,         # Taxa de crossover
    'mutpb': 0.2,        # Taxa de mutação
}

# Executar otimização
resultado = run_ga_experiment(**experimento)
print(f"Melhor F1-score: {resultado['f1_test']:.4f}")
```

## 📈 Visualizações

O projeto inclui visualizações para:

- 📊 **Evolução do fitness** ao longo das gerações
- 🔄 **Comparação de performance** entre configurações
- ⏰ **Análise custo-benefício** (tempo vs melhoria)
- 🎯 **Precisão vs Recall** para cada experimento
- 🧬 **Parâmetros finais** descobertos pelos GAs

## 🔍 Insights Principais

### Quando Usar GAs
✅ **Recomendado quando:**
- Performance do modelo é crítica
- Recursos computacionais são adequados
- Busca exaustiva é inviável
- Soluções inovadoras são desejadas

❌ **Não recomendado quando:**
- Tempo é limitado (prototipagem)
- Recursos computacionais restritos
- Baseline já atende requisitos

### Configurações Recomendadas

| Cenário | População | Gerações | Tempo Esperado |
|---------|-----------|----------|----------------|
| **Desenvolvimento rápido** | 20-30 | 15-25 | 10-20s |
| **Produção equilibrada** | 40-50 | 30-50 | 30-60s |
| **Performance máxima** | 60-80 | 40-60 | 60-120s |

## 🎓 Fundamentação Teórica

### Algoritmos Genéticos
- Inspirados na evolução biológica
- Operadores: seleção, crossover, mutação
- Exploração inteligente do espaço de busca
- Robustos a ótimos locais

### Hyperparameter Tuning
- Grid Search vs Random Search vs Métodos Avançados
- Importância da validação cruzada
- Trade-off exploration vs exploitation

## 📚 Referências Principais

1. **Bergstra, J., & Bengio, Y.** (2012). Random search for hyper-parameter optimization. *JMLR*.
2. **Holland, J. H.** (1992). Adaptation in natural and artificial systems. *MIT Press*.
3. **Eiben, A. E., & Smith, J. E.** (2015). Introduction to evolutionary computing. *Springer*.
4. **Snoek, J., et al.** (2012). Practical bayesian optimization of machine learning algorithms. *NIPS*.
5. **Feurer, M., & Hutter, F.** (2019). Hyperparameter optimization. *Automated ML*.

## 🔧 Tecnologias Utilizadas

- **Python 3.7+**: Linguagem principal
- **scikit-learn**: MLPClassifier e métricas
- **DEAP**: Framework de algoritmos evolutivos
- **NumPy**: Operações numéricas
- **Pandas**: Manipulação de dados
- **Matplotlib**: Visualizações

## 📈 Resultados Detalhados

### Métricas de Performance

| Configuração | F1-Score | Precisão | Recall | Tempo (s) | Melhoria |
|--------------|----------|----------|--------|-----------|----------|
| Baseline | 0.8654 | 0.8523 | 0.8789 | 2.3 | - |
| GA_Config1 | 0.8923 | 0.8834 | 0.9015 | 15.7 | +3.1% |
| GA_Config2 | 0.9087 | 0.8978 | 0.9201 | 42.1 | +5.0% |
| GA_Config3 | **0.9134** | **0.9023** | **0.9248** | 67.8 | **+5.6%** |
| GA_Config4 | 0.8998 | 0.8889 | 0.9112 | 58.3 | +4.0% |

## 🚧 Limitações e Trabalhos Futuros

### Limitações Identificadas
- Dataset sintético de complexidade limitada
- Espaço de hiperparâmetros restrito
- Análise de execução única (sem robustez estatística)
- Ausência de comparação com outros métodos

### Trabalhos Futuros
- Validação em datasets reais complexos
- Comparação com Bayesian Optimization
- Análise estatística com múltiplas execuções
- Otimização multi-objetivo
- Paralelização dos algoritmos


## 📄 Licença

### **Reconhecimentos e Direitos Autorais**

**@autor:** ¹Euderlan Freire - ²Hissa Bárbara - ³Lucas Silva - Maria Clara. 
**@contato:** [³computer.lucas2@gmail.com]  
**@data última versão:** 12 de junho de 2025  
**@versão:** 1.0  
**@outros repositórios:** [URLs - apontem para os seus Gits AQUI]  
**@Agradecimentos:** Universidade Federal do Maranhão (UFMA), Professor Doutor Thales Levi Azevedo Valente, e colegas de curso.

### **Copyright/License**

Este material é resultado de um trabalho acadêmico para a disciplina **INTELIGÊNCIA ARTIFICIAL**, sob a orientação do professor **Dr. THALES LEVI AZEVEDO VALENTE**, semestre letivo **2025.1**, curso **Engenharia da Computação**, na **Universidade Federal do Maranhão (UFMA)**.

Todo o material sob esta licença é software livre: pode ser usado para fins acadêmicos e comerciais sem nenhum custo. Não há papelada, nem royalties, nem restrições de "copyleft" do tipo GNU. Ele é licenciado sob os termos da **Licença MIT**, conforme descrito abaixo, e, portanto, é compatível com a GPL e também se qualifica como software de código aberto. É de domínio público. Os detalhes legais estão abaixo. O espírito desta licença é que você é livre para usar este material para qualquer finalidade, sem nenhum custo. O único requisito é que, se você usá-los, nos dê crédito.

### **Licença MIT**

Licenciado sob a Licença MIT. Permissão é concedida, gratuitamente, a qualquer pessoa que obtenha uma cópia deste software e dos arquivos de documentação associados (o "Software"), para lidar no Software sem restrição, incluindo sem limitação os direitos de usar, copiar, modificar, mesclar, publicar, distribuir, sublicenciar e/ou vender cópias do Software, e permitir pessoas a quem o Software é fornecido a fazê-lo, sujeito às seguintes condições:

Este aviso de direitos autorais e este aviso de permissão devem ser incluídos em todas as cópias ou partes substanciais do Software.

**O SOFTWARE É FORNECIDO "COMO ESTÁ", SEM GARANTIA DE QUALQUER TIPO, EXPRESSA OU IMPLÍCITA, INCLUINDO MAS NÃO SE LIMITANDO ÀS GARANTIAS DE COMERCIALIZAÇÃO, ADEQUAÇÃO A UM DETERMINADO FIM E NÃO INFRINGÊNCIA. EM NENHUM CASO OS AUTORES OU DETENTORES DE DIREITOS AUTORAIS SERÃO RESPONSÁVEIS POR QUALQUER RECLAMAÇÃO, DANOS OU OUTRA RESPONSABILIDADE, SEJA EM AÇÃO DE CONTRATO, TORT OU OUTRA FORMA, DECORRENTE DE, FORA DE OU EM CONEXÃO COM O SOFTWARE OU O USO OU OUTRAS NEGOCIAÇÕES NO SOFTWARE.**

Para mais informações sobre a Licença MIT: https://opensource.org/licenses/MIT

## 📞 Contato

**Autor**: [EUderlan Freire]
- 📧 Email: [euderlan.freire@discente.ufma.br]




**⭐ Se este projeto foi útil, considere dar uma estrela!**

</div>
