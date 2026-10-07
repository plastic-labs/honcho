import { expect, test } from "bun:test"

import { __testing } from "../dist/index.js"

test("root sessions model the user and leave agent observeMe off by default", () => {
  const topology = __testing.buildPeerTopology({
    config: {},
    userPeerId: "user",
    rootAgentPeerId: "kilo",
    activeAgentPeerId: "kilo",
    childAgentPeerId: null,
    parentAgentObserverPeerId: null,
  })

  expect(topology.sessionPeerConfigs).toEqual({
    user: { observeMe: true, observeOthers: false },
    kilo: { observeMe: false, observeOthers: true },
  })
})

test("agentObserveMe true turns on self-observation on the root agent peer", () => {
  const topology = __testing.buildPeerTopology({
    config: { agentObserveMe: true },
    userPeerId: "user",
    rootAgentPeerId: "kilo",
    activeAgentPeerId: "kilo",
    childAgentPeerId: null,
    parentAgentObserverPeerId: null,
  })

  expect(topology.sessionPeerConfigs.kilo.observeMe).toBe(true)
})
