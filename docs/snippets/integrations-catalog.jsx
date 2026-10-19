// Integrations catalog for /v3/guides/overview.
// Order inside each section is priority order — keep it that way when adding entries.
// status: "official" | "community"
// icon: a Font Awesome name, or a pinned CDN URL to a monochrome black SVG
// (Simple Icons, then Lobe Icons) on jsDelivr; style.css inverts them in dark mode.

export const IntegrationsCatalog = () => {
  const featured = [
    {
      name: "Hermes Agent",
      icon: "https://cdn.jsdelivr.net/npm/@lobehub/icons-static-svg@1.95.1/icons/hermesagent.svg",
      href: "/v3/guides/integrations/hermes",
      cta: "Set up Hermes",
      desc: "Cross-session memory and user modeling for Nous Research's Hermes agent, across Telegram, Discord, Slack, and WhatsApp.",
    },
    {
      name: "OpenClaw",
      icon: "https://cdn.jsdelivr.net/npm/@lobehub/icons-static-svg@1.95.1/icons/openclaw.svg",
      href: "/v3/guides/integrations/openclaw",
      cta: "Set up OpenClaw",
      desc: "Memory across every OpenClaw channel, with honcho_* tools and automatic message capture.",
    },
    {
      name: "Coding Agents",
      icon: "square-terminal",
      href: "#coding-agents",
      cta: "Pick your agent",
      desc: "Long-term memory for Claude Code, Codex, OpenCode, Kilo Code, Pi, and more — preferences and project context that survive context wipes.",
    },
  ];

  const universal = [
    {
      name: "MCP Server",
      icon: "https://cdn.jsdelivr.net/npm/simple-icons@16.33.0/icons/modelcontextprotocol.svg",
      href: "/v3/guides/integrations/mcp",
      desc: "Any MCP client — Claude Desktop, Cursor, Windsurf, VS Code, Zed, Goose, Cline.",
    },
  ];

  const sections = [
    {
      id: "general-agents",
      title: "General agents",
      blurb: "Personal agents that work across your channels and tools.",
      items: [
        { name: "Hermes Agent", icon: "https://cdn.jsdelivr.net/npm/@lobehub/icons-static-svg@1.95.1/icons/hermesagent.svg", href: "/v3/guides/integrations/hermes", desc: "Cross-session memory and user modeling for Nous Research's Hermes agent.", status: "official" },
        { name: "OpenClaw", icon: "https://cdn.jsdelivr.net/npm/@lobehub/icons-static-svg@1.95.1/icons/openclaw.svg", href: "/v3/guides/integrations/openclaw", desc: "Memory across WhatsApp, Telegram, Discord, Slack, and every other OpenClaw channel.", status: "official" },
        { name: "Claude Desktop", icon: "https://cdn.jsdelivr.net/npm/simple-icons@16.33.0/icons/claude.svg", href: "/v3/guides/integrations/mcp#claude-desktop", desc: "Give Claude Desktop memory of you across every chat.", status: "official", via: "MCP" },
        { name: "Goose", icon: "https://cdn.jsdelivr.net/npm/@lobehub/icons-static-svg@1.95.1/icons/goose.svg", href: "/v3/guides/integrations/mcp#goose", desc: "Add Honcho to Goose as a remote extension.", status: "official", via: "MCP" },
      ],
    },
    {
      id: "coding-agents",
      title: "Coding agents",
      blurb: "Memory that carries across sessions, restarts, and projects in the coding tools you already use.",
      items: [
        { name: "Claude Code", icon: "https://cdn.jsdelivr.net/npm/simple-icons@16.33.0/icons/claudecode.svg", href: "/v3/guides/integrations/claude-code", desc: "Memory that survives context wipes, restarts, and project switches.", status: "official" },
        { name: "Pi", icon: "https://cdn.jsdelivr.net/npm/simple-icons@16.33.0/icons/pi.svg", href: "/v3/guides/community/pi-honcho-memory", desc: "Persistent memory extension for the pi coding agent CLI.", status: "community" },
        { name: "Codex", icon: "https://cdn.jsdelivr.net/npm/simple-icons@15.22.0/icons/openai.svg", href: "/v3/guides/integrations/codex", desc: "Lifecycle hooks capture Codex sessions and restore context on start.", status: "official" },
        { name: "OpenCode", icon: "https://cdn.jsdelivr.net/npm/simple-icons@16.33.0/icons/opencode.svg", href: "/v3/guides/integrations/opencode", desc: "Per-directory, per-repo, or branch-scoped session memory.", status: "official" },
        { name: "Kilo Code", icon: "https://cdn.jsdelivr.net/npm/@lobehub/icons-static-svg@1.95.1/icons/kilocode.svg", href: "/v3/guides/integrations/kilo", desc: "Memory across the Kilo CLI, VS Code, and JetBrains from one install.", status: "official" },
        { name: "DeepSeek Harness", icon: "https://cdn.jsdelivr.net/npm/simple-icons@16.33.0/icons/deepseek.svg", href: "/v3/guides/integrations/deepseek-harness", desc: "Context injection, turn capture, and honcho_search for dsh.", status: "official" },
        { name: "Cline", icon: "https://cdn.jsdelivr.net/npm/simple-icons@16.33.0/icons/cline.svg", href: "/v3/guides/integrations/mcp#cline", desc: "Connect Cline to the Honcho MCP server.", status: "official", via: "MCP" },
        { name: "Cursor", icon: "https://cdn.jsdelivr.net/npm/simple-icons@16.33.0/icons/cursor.svg", href: "/v3/guides/integrations/mcp#cursor", desc: "Add Honcho as an HTTP MCP server in Cursor's global or per-project config.", status: "official", via: "MCP" },
        { name: "Windsurf", icon: "https://cdn.jsdelivr.net/npm/simple-icons@16.33.0/icons/windsurf.svg", href: "/v3/guides/integrations/mcp#windsurf", desc: "Give Windsurf's Cascade agent memory through the Honcho MCP server.", status: "official", via: "MCP" },
        { name: "VS Code", icon: "https://cdn.jsdelivr.net/npm/simple-icons@12.4.0/icons/visualstudiocode.svg", href: "/v3/guides/integrations/mcp#vs-code-copilot-chat", desc: "Memory for GitHub Copilot Chat in VS Code.", status: "official", via: "MCP" },
        { name: "Zed", icon: "https://cdn.jsdelivr.net/npm/simple-icons@16.33.0/icons/zedindustries.svg", href: "/v3/guides/integrations/mcp#zed", desc: "Add Honcho to Zed as a context server.", status: "official", via: "MCP" },
      ],
    },
    {
      id: "frameworks",
      title: "Frameworks & SDKs",
      blurb: "Add Honcho to agents built on these frameworks.",
      items: [
        { name: "Vercel AI SDK", icon: "https://cdn.jsdelivr.net/npm/simple-icons@16.33.0/icons/vercel.svg", href: "/v3/guides/integrations/vercel-ai-sdk", desc: "Memory middleware and tools for generateText and streamText.", status: "official" },
        { name: "LangGraph", icon: "https://cdn.jsdelivr.net/npm/simple-icons@16.33.0/icons/langgraph.svg", href: "/v3/guides/integrations/langgraph", desc: "Persistent memory and theory of mind for LangGraph agents.", status: "official" },
        { name: "CrewAI", icon: "https://cdn.jsdelivr.net/npm/simple-icons@16.33.0/icons/crewai.svg", href: "/v3/guides/integrations/crewai", desc: "Honcho as a storage backend for CrewAI's Memory API.", status: "official" },
        { name: "Agent Zero", icon: "https://cdn.jsdelivr.net/gh/agent0ai/agent-zero@e3051fb584b1a36be2b0a0c90606f1c2c2d356ec/webui/public/darkSymbol.svg", href: "/v3/guides/community/agent0", desc: "Persistent memory plugin for the Agent Zero framework.", status: "community" },
        { name: "Paperclip", icon: "https://cdn.jsdelivr.net/npm/lucide-static@1.50.0/icons/paperclip.svg", href: "/v3/guides/integrations/paperclip", desc: "Memory for Paperclip companies, agents, issues, and documents.", status: "official" },
      ],
    },
    {
      id: "apps",
      title: "Apps & workflows",
      blurb: "Memory for chat apps and workflow builders.",
      items: [
        { name: "SillyTavern", icon: "comments", href: "/v3/guides/integrations/sillytavern", desc: "Long-term memory for SillyTavern characters.", status: "official" },
        { name: "n8n", icon: "https://cdn.jsdelivr.net/npm/simple-icons@16.33.0/icons/n8n.svg", href: "/v3/guides/integrations/n8n", desc: "Importable workflow for memory-aware n8n automations.", status: "official" },
        { name: "Zo Computer", icon: "bolt", href: "/v3/guides/integrations/zo-computer", desc: "Persistent memory skill for Zo workflows.", status: "official" },
      ],
    },
    {
      id: "data-sources",
      title: "Data sources",
      blurb: "Import data from the products you already use into Honcho.",
      items: [
        { name: "Gmail", icon: "https://cdn.jsdelivr.net/npm/simple-icons@16.33.0/icons/gmail.svg", href: "/v3/guides/gmail", desc: "Import email threads as peers, sessions, and messages.", status: "official" },
        { name: "Granola", icon: "https://cdn.jsdelivr.net/gh/pheralb/svgl@ed75393dbe6eba6e446e208abb6826ecd1abd36d/static/library/granola-light.svg", href: "/v3/guides/granola", desc: "Ingest meeting transcripts with speaker turns and participants.", status: "official" },
      ],
    },
  ];

  const filters = [
    { key: "all", label: "All" },
    { key: "official", label: "Official" },
    { key: "community", label: "Community" },
  ];

  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("all");

  const q = query.trim().toLowerCase();
  const matches = (item) =>
    (filter === "all" || item.status === filter) &&
    (!q || (item.name + " " + item.desc).toLowerCase().includes(q));
  const visible = sections
    .map((s) => ({ ...s, items: s.items.filter(matches) }))
    .filter((s) => s.items.length > 0);
  const total = sections.reduce((n, s) => n + s.items.length, 0);
  const shown = visible.reduce((n, s) => n + s.items.length, 0);
  const browsing = !q && filter === "all";

  const pill = (status, via) => {
    if (status === "community") {
      return <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 dark:bg-amber-400/10 dark:text-amber-300">Community</span>;
    }
    if (via) {
      return <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-primary/10 text-primary dark:text-primary-light">via {via}</span>;
    }
    return null;
  };

  const ItemCard = ({ item }) => (
    <a href={item.href} className="block h-full border-none no-underline">
      <div className="h-full flex flex-col gap-1.5 rounded-xl border border-zinc-950/10 dark:border-white/10 p-4 transition-colors hover:border-primary dark:hover:border-primary-light">
        <div className="flex items-center gap-2.5">
          <Icon icon={item.icon} size={18} />
          <span className="font-semibold text-zinc-950 dark:text-white">{item.name}</span>
          <span className="ml-auto">{pill(item.status, item.via)}</span>
        </div>
        <p className="m-0 text-sm leading-snug text-zinc-600 dark:text-zinc-400">{item.desc}</p>
      </div>
    </a>
  );

  return (
    <div className="not-prose">
      {browsing && (
        <div className="mb-8">
          <div className="mb-3 text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">Works with any agent</div>
          <div className="grid grid-cols-1 gap-3">
            {universal.map((u) => (
              <a key={u.name} href={u.href} className="block border-none no-underline">
                <div className="h-full flex items-start gap-3 rounded-xl border border-zinc-950/10 dark:border-white/10 p-4 transition-colors hover:border-primary dark:hover:border-primary-light">
                  <div className="flex items-center justify-center w-8 h-8 shrink-0 rounded-lg bg-primary/10">
                    <Icon icon={u.icon} size={16} color="#66AAFF" />
                  </div>
                  <div>
                    <div className="font-semibold text-zinc-950 dark:text-white">{u.name}</div>
                    <p className="m-0 mt-0.5 text-sm leading-snug text-zinc-600 dark:text-zinc-400">{u.desc}</p>
                  </div>
                </div>
              </a>
            ))}
          </div>
        </div>
      )}
      {browsing && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-10">
          {featured.map((f) => (
            <a key={f.name} href={f.href} className="group block border-none no-underline">
              <div className="h-full flex flex-col gap-3 rounded-2xl border border-zinc-950/10 dark:border-white/10 p-5 transition-colors group-hover:border-primary dark:group-hover:border-primary-light">
                <div className="flex items-center gap-3">
                  <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-primary/10">
                    <Icon icon={f.icon} size={18} color="#66AAFF" />
                  </div>
                  <span className="text-lg font-semibold text-zinc-950 dark:text-white">{f.name}</span>
                </div>
                <p className="m-0 text-sm leading-relaxed text-zinc-600 dark:text-zinc-400">{f.desc}</p>
                <span className="mt-auto text-sm font-medium text-primary dark:text-primary-light">{f.cta} →</span>
              </div>
            </a>
          ))}
        </div>
      )}

      <div className="flex flex-col sm:flex-row sm:items-center gap-3 mb-8">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search integrations…"
          className="flex-1 rounded-lg border border-zinc-950/10 dark:border-white/10 bg-transparent px-3 py-2 text-sm text-zinc-950 dark:text-white outline-none focus:border-primary"
        />
        <div className="flex flex-wrap items-center gap-1.5">
          {filters.map((f) => (
            <button
              key={f.key}
              type="button"
              onClick={() => setFilter(f.key)}
              className={"text-xs font-medium px-3 py-1.5 rounded-full border " + (filter === f.key ? "border-primary bg-primary/10 text-primary dark:text-primary-light" : "border-zinc-950/10 dark:border-white/10 text-zinc-600 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-white")}
            >
              {f.label}
            </button>
          ))}
          <span className="ml-1 text-xs text-zinc-500">{browsing ? total + " available" : shown + " shown"}</span>
        </div>
      </div>

      {visible.length === 0 && (
        <p className="text-sm text-zinc-500">
          Nothing matches “{query}”. Any MCP client can use the <a href="/v3/guides/integrations/mcp" className="text-primary">Honcho MCP server</a>.
        </p>
      )}

      {visible.map((s) => (
        <section key={s.id} id={s.id} className="mb-10 scroll-mt-24">
          <div className="flex items-baseline gap-2 mb-1">
            <h2 className="m-0 text-xl font-semibold text-zinc-950 dark:text-white">{s.title}</h2>
            <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-zinc-100 text-zinc-500 dark:bg-white/5 dark:text-zinc-400">{s.items.length}</span>
          </div>
          <p className="mt-0 mb-4 text-sm text-zinc-500 dark:text-zinc-400">{s.blurb}</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {s.items.map((item) => (
              <ItemCard key={item.name} item={item} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
};
