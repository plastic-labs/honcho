import { describe, expect, test } from 'bun:test'
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const pkgRoot = join(import.meta.dir, '..')

function run(cmd: string[], cwd: string) {
  const p = Bun.spawnSync(cmd, { cwd, stdout: 'pipe', stderr: 'pipe' })
  return { code: p.exitCode, out: p.stdout.toString() + p.stderr.toString() }
}

describe('published package', () => {
  test('the npm tarball loads under plain Node and compiles under tsc nodenext', () => {
    // Build + pack exactly what `npm publish` would ship.
    expect(run(['bun', 'run', 'build'], pkgRoot).code).toBe(0)
    const consumer = mkdtempSync(join(tmpdir(), 'hpc-consumer-'))
    const pack = run(['npm', 'pack', '--pack-destination', consumer, '--json'], pkgRoot)
    expect(pack.code).toBe(0)
    const tarball = join(consumer, JSON.parse(pack.out)[0].filename)

    writeFileSync(join(consumer, 'package.json'), JSON.stringify({ name: 'consumer', type: 'module', private: true }))
    const dir = join(consumer, 'node_modules', '@honcho-ai', 'harness-plugin-core')
    mkdirSync(dir, { recursive: true })
    expect(run(['tar', 'xzf', tarball, '--strip-components=1', '-C', dir], consumer).code).toBe(0)

    const source =
      'import { hostHeaderValue } from "@honcho-ai/harness-plugin-core"\nconsole.log(hostHeaderValue({ host: "test", hostVersion: "1.0.0", platform: "linux" }))\n'

    // Plain-JS Node: no TypeScript transpiler in the loop.
    writeFileSync(join(consumer, 'main.mjs'), source)
    const node = run(['node', 'main.mjs'], consumer)
    expect(node.out.trim()).toBe("test/1.0.0 (linux)")
    expect(node.code).toBe(0)

    // tsc with nodenext resolution reads the shipped .d.ts, not our source.
    writeFileSync(join(consumer, 'main.ts'), source)
    writeFileSync(
      join(consumer, 'tsconfig.json'),
      JSON.stringify({
        compilerOptions: { module: 'NodeNext', moduleResolution: 'NodeNext', target: 'ES2022', strict: true, noEmit: true },
        include: ['main.ts'],
      }),
    )
    const tsc = run([join(pkgRoot, 'node_modules', '.bin', 'tsc'), '-p', 'tsconfig.json'], consumer)
    expect(tsc.out.trim()).toBe('')
    expect(tsc.code).toBe(0)
  })
})
