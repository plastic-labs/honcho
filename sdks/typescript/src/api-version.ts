/**
 * API version used for all Honcho API requests.
 */
export const API_VERSION = 'v3'

/**
 * This SDK's version, sent to Honcho in `X-Honcho-Host`. Kept in sync with
 * package.json by scripts/update_version.py; importing package.json instead would
 * make bundlers inline the whole manifest into shipped code.
 */
export const VERSION = '2.4.0'
