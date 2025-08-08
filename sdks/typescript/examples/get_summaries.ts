/**
 * Example of getting session summaries using the Honcho TypeScript SDK.
 *
 * This example demonstrates how to retrieve both short and long summaries
 * for a session, including their metadata like message ID, creation timestamp,
 * and token count.
 */

import { Honcho, SessionSummaries } from '@honcho-ai/honcho'

async function main() {
  // Initialize the Honcho client
  const apiKey = process.env.HONCHO_API_KEY
  if (!apiKey) {
    throw new Error('Please set HONCHO_API_KEY environment variable')
  }

  const client = new Honcho({ apiKey })

  // Get a session (replace with your actual session ID)
  const session = await client.session('my-conversation-session')

  // Get summaries for the session
  const summaries: SessionSummaries = await session.getSummaries()

  console.log(`Session ID: ${summaries.id}`)
  console.log('-'.repeat(50))

  // Check and display short summary
  if (summaries.shortSummary) {
    console.log('SHORT SUMMARY:')
    console.log(`  Content: ${summaries.shortSummary.content.slice(0, 200)}...`) // First 200 chars
    console.log(`  Covers up to message ID: ${summaries.shortSummary.messageId}`)
    console.log(`  Created at: ${summaries.shortSummary.createdAt}`)
    console.log(`  Token count: ${summaries.shortSummary.tokenCount}`)
    console.log(`  Type: ${summaries.shortSummary.summaryType}`)
  } else {
    console.log('No short summary available yet')
  }

  console.log('-'.repeat(50))

  // Check and display long summary
  if (summaries.longSummary) {
    console.log('LONG SUMMARY:')
    console.log(`  Content: ${summaries.longSummary.content.slice(0, 200)}...`) // First 200 chars
    console.log(`  Covers up to message ID: ${summaries.longSummary.messageId}`)
    console.log(`  Created at: ${summaries.longSummary.createdAt}`)
    console.log(`  Token count: ${summaries.longSummary.tokenCount}`)
    console.log(`  Type: ${summaries.longSummary.summaryType}`)
  } else {
    console.log('No long summary available yet')
  }

  // Example showing how to use summaries in application logic
  console.log('\n' + '='.repeat(50))
  console.log('USAGE EXAMPLE:')
  console.log('='.repeat(50))

  // Check if we have any summaries
  if (summaries.shortSummary || summaries.longSummary) {
    // Prefer long summary if available and under token limit
    const MAX_TOKENS = 2000
    let selectedSummary = null

    if (
      summaries.longSummary &&
      summaries.longSummary.tokenCount <= MAX_TOKENS
    ) {
      selectedSummary = summaries.longSummary
      console.log('Using long summary (within token limit)')
    } else if (summaries.shortSummary) {
      selectedSummary = summaries.shortSummary
      console.log('Using short summary')
    }

    if (selectedSummary) {
      console.log(`Selected summary covers up to message ${selectedSummary.messageId}`)
      console.log(`Token count: ${selectedSummary.tokenCount}`)

      // You could use this summary as context for an LLM call
      // For example, with OpenAI:
      // const systemPrompt = `Previous conversation summary: ${selectedSummary.content}`
    }
  } else {
    console.log('No summaries available - would need to use full message history')
  }
}

// Run the example
main().catch(console.error)
