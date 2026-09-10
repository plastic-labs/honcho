// Markdown files are bundled as text strings via the wrangler `Text` rule.
declare module "*.md" {
  const content: string;
  export default content;
}
