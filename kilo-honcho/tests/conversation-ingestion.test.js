import { expect, test } from "bun:test"

import { __testing } from "../dist/index.js"

test("extractCompletedAssistantMessage uses current-turn parts and completion timestamp", () => {
  const partState = new Map()

  __testing.upsertAssistantMessagePart(partState, {
    event: {
      type: "message.part.updated",
      properties: {
        part: {
          id: "part-reasoning",
          sessionID: "ses-123",
          messageID: "msg-123",
          type: "reasoning",
          text: "internal",
          time: { start: 1710000000000 },
        },
      },
    },
  })
  __testing.upsertAssistantMessagePart(partState, {
    event: {
      type: "message.part.updated",
      properties: {
        part: {
          id: "part-visible",
          sessionID: "ses-123",
          messageID: "msg-123",
          type: "text",
          text: "Final reply",
          time: { start: 1710000000001 },
        },
      },
    },
  })

  expect(
    __testing.extractCompletedAssistantMessage(
      {
        event: {
          type: "message.updated",
          properties: {
            info: {
              id: "msg-123",
              sessionID: "ses-123",
              role: "assistant",
              time: {
                created: 1710000000000,
                completed: 1710000004321,
              },
            },
          },
        },
      },
      partState,
    ),
  ).toEqual({
    messageId: "msg-123",
    sessionID: "ses-123",
    text: "Final reply",
    createdAt: "2024-03-09T16:00:04.321Z",
  })
})
