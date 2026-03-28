# 🎨 UI-UX Pro Max Skill — Guia de Instalação e Uso

> Skill de design system com IA para Claude Code e outros assistentes de código.
> Versão v2.0 | Repositório: [nextlevelbuilder/ui-ux-pro-max-skill](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill)

---

## 📦 Instalação (via CLI — Recomendado)

Você **não precisa clonar o repositório**. A instalação é feita via CLI global:

```bash
# 1. Instalar o CLI globalmente
npm install -g uipro-cli

# 2. Entrar na raiz do seu projeto
cd /caminho/do/seu/projeto

# 3. Inicializar a skill para o seu assistente
uipro init --ai claude       # Claude Code
uipro init --ai cursor       # Cursor
uipro init --ai windsurf     # Windsurf
uipro init --ai copilot      # GitHub Copilot
uipro init --ai all          # Todos os assistentes
```

### Instalação Global (disponível para todos os projetos)

```bash
uipro init --ai claude --global   # Instala em ~/.claude/skills/
uipro init --ai cursor --global   # Instala em ~/.cursor/skills/
```

### Instalação Offline

```bash
uipro init --offline   # Usa assets locais, sem download do GitHub
```

---

## ✅ Pré-requisitos

- **Node.js** (para o CLI `uipro`)
- **Python 3.x** (para o motor de busca/design system)

```bash
# Verificar Python
python3 --version

# Instalar Python se necessário
brew install python3          # macOS
sudo apt install python3      # Ubuntu/Debian
winget install Python.Python.3.12  # Windows
```

---

## 🚀 Como Usar

### Modo Skill (ativação automática — Claude Code, Cursor, Windsurf...)

Basta pedir normalmente no chat:

```
Build a landing page for my SaaS product
Create a dashboard for healthcare analytics
Design a portfolio website with dark mode
Make a mobile app UI for e-commerce
```

A skill ativa automaticamente para qualquer solicitação de UI/UX.

### Modo Workflow (slash command — Kiro, GitHub Copilot, Roo Code)

```
/ui-ux-pro-max Build a landing page for my SaaS product
```

---

## ⚙️ Comando Avançado — Design System Generator

```bash
# Gerar design system com saída ASCII
python3 .claude/skills/ui-ux-pro-max/scripts/search.py "beauty spa wellness" --design-system -p "Serenity Spa"

# Gerar com saída Markdown
python3 .claude/skills/ui-ux-pro-max/scripts/search.py "fintech banking" --design-system -f markdown

# Busca por domínio específico
python3 .claude/skills/ui-ux-pro-max/scripts/search.py "glassmorphism" --domain style
python3 .claude/skills/ui-ux-pro-max/scripts/search.py "elegant serif" --domain typography
python3 .claude/skills/ui-ux-pro-max/scripts/search.py "dashboard" --domain chart

# Diretrizes por stack
python3 .claude/skills/ui-ux-pro-max/scripts/search.py "form validation" --stack react
python3 .claude/skills/ui-ux-pro-max/scripts/search.py "responsive layout" --stack html-tailwind
```

---

## 💾 Persistir Design System (padrão Master + Overrides)

Salva o design system em arquivos para reutilização entre sessões:

```bash
# Gerar e persistir em design-system/MASTER.md
python3 .claude/skills/ui-ux-pro-max/scripts/search.py "SaaS dashboard" --design-system --persist -p "MyApp"

# Criar override para uma página específica
python3 .claude/skills/ui-ux-pro-max/scripts/search.py "SaaS dashboard" --design-system --persist -p "MyApp" --page "dashboard"
```

### Estrutura gerada:

```
design-system/
├── MASTER.md            # Fonte da verdade global (cores, tipografia, espaçamento, componentes)
└── pages/
    └── dashboard.md     # Overrides por página (apenas desvios do Master)
```

### Prompt para recuperação contextual:

```
I am building the [Page Name] page. Please read design-system/MASTER.md.
Also check if design-system/pages/[page-name].md exists.
If the page file exists, prioritize its rules.
If not, use the Master rules exclusively.
Now, generate the code...
```

---

## 🛠️ Outros Comandos CLI

```bash
uipro versions         # Listar versões disponíveis
uipro update           # Atualizar para a versão mais recente
uipro uninstall        # Remover skill (detecta plataforma automaticamente)
uipro uninstall --ai claude   # Remover plataforma específica
uipro uninstall --global      # Remover instalação global
```

---

## 🗂️ Stacks Suportadas

| Categoria       | Stacks                                      |
|----------------|----------------------------------------------|
| Web (HTML)      | HTML + Tailwind (padrão)                    |
| React           | React, Next.js, shadcn/ui                   |
| Vue             | Vue, Nuxt.js, Nuxt UI                       |
| Angular         | Angular                                     |
| PHP             | Laravel (Blade, Livewire, Inertia.js)       |
| Outros Web      | Svelte, Astro                               |
| iOS             | SwiftUI                                     |
| Android         | Jetpack Compose                             |
| Cross-Platform  | React Native, Flutter                       |

---

## 🎨 O que a Skill Oferece

| Recurso                  | Quantidade |
|--------------------------|------------|
| Estilos de UI            | 67         |
| Paletas de cores         | 161        |
| Combinações de fontes    | 57         |
| Tipos de gráficos        | 25         |
| Stacks suportadas        | 15         |
| Diretrizes UX            | 99         |
| Regras de raciocínio     | 161        |

---

## ❌ Quando NÃO Clonar o Repositório

A clonagem do repositório é **apenas para contribuidores**. Para uso normal em projetos, use sempre o CLI:

```bash
npm install -g uipro-cli
uipro init --ai claude
```

Isso garante a versão mais recente com a estrutura correta de arquivos para o seu assistente.

---

## 🤝 Contribuindo (opcional)

```bash
git clone https://github.com/nextlevelbuilder/ui-ux-pro-max-skill.git
cd ui-ux-pro-max-skill

# Estrutura principal
# src/ui-ux-pro-max/data/*.csv       → Banco de dados
# src/ui-ux-pro-max/scripts/*.py     → Motor de busca
# src/ui-ux-pro-max/templates/       → Templates por plataforma
# cli/                               → CLI installer

# Sincronizar e testar localmente
cp -r src/ui-ux-pro-max/data/* cli/assets/data/
cp -r src/ui-ux-pro-max/scripts/* cli/assets/scripts/
cd cli && bun run build
node dist/index.js init --ai claude --offline

# Criar PR
git checkout -b feat/sua-feature
git commit -m "feat: descrição"
git push -u origin feat/sua-feature
gh pr create
```