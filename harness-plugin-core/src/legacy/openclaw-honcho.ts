import type { LegacySettings } from '../config.js'

/**
 * openclaw-honcho up to 1.7.0 read nothing from this file. Its key, server and
 * workspace came from OpenClaw's own plugin config or `HONCHO_*` (default
 * workspace `openclaw`), and its user peer was always `owner`. The schema-2
 * plugin still reads OpenClaw's config first, so only the peer needs keeping:
 * this stops a top-level `peerName` or `$USER` from moving its users off `owner`.
 */
export function openclawHonchoReader(): LegacySettings {
  return { peerName: 'owner' }
}
