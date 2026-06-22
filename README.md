# INF01090-Final
Trabalho final da disciplina de Ciência de Dados (INF01090) — Instituto de Informática (INF/UFRGS).

## Como Executar e Desenvolver (via Nix Flakes)

Para garantir a reprodutibilidade e a execução idêntica em qualquer máquina, o projeto utiliza o ecossistema **Nix**.

### 1. Pré-requisitos
Certifique-se de ter o gerenciador de pacotes **Nix** instalado na sua máquina com o suporte a `flakes` e `nix-command` ativos. Utilize o comando correspondente ao seu sistema operacional para realizar a instalação:

* **Linux (Geral) e WSL:**
    ```bash
    curl --proto '=https' --tlsv1.2 -L [https://nixos.org/nix/install](https://nixos.org/nix/install) | sh -s -- --daemon
    ```
* **macOS:**
    ```bash
    curl --proto '=https' --tlsv1.2 -L [https://nixos.org/nix/install](https://nixos.org/nix/install) | sh
    ```

*(Se você já utiliza o **NixOS**, seu sistema já atende nativamente a este requisito).*

### 2. Entrando no Ambiente de Desenvolvimento
Navegue até a raiz do projeto e inicialize o *development shell* declarado no arquivo `flake.nix`. O Nix irá baixar, isolar e disponibilizar automaticamente todas as dependências necessárias (interpretador Python e bibliotecas):

```bash
cd INF01090-Final/
nix develop
```

### 3. Integração com o VS Code (para notebook)
Caso utilize o VS Code, você pode integrar o ambiente isolando os runtimes através de extensões como a `Nix Environment Selector` ou `direnv`. 
Certifique-se de selecionar o kernel do .venv gerado pelo shell do Nix ao rodar os notebooks (.ipynb).