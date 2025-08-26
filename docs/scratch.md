# Honcho Documentation Fix - TypeScript Constructor and Demo Server Warning

## Date: 2025-08-26
## Author: Claude (with Eri)

## Summary
Fixed TypeScript SDK initialization to use `new Honcho({})` instead of `new Honcho()` and added clear warnings about demo server defaults.

## Changes Made

### 1. TypeScript Constructor Fix
**Issue:** TypeScript SDK requires an options object parameter, even if empty. `new Honcho()` causes a compilation error.

**Files Modified:**
- `/v2/documentation/introduction/quickstart.mdx` - Line 64
- `/v2/guides/streaming-response.mdx` - Lines 53, 181
- `/v2/guides/dialectic-endpoint.mdx` - Line 41

**Change:** 
```typescript
// Before (incorrect)
const client = new Honcho();

// After (correct)  
const client = new Honcho({});
```

### 2. Demo Server Warning in Quickstart
**Location:** `/v2/documentation/introduction/quickstart.mdx` - Lines 52-65

**Added:**
- Clear note that default is demo server
- Link to sign up at app.honcho.dev
- Direct link to API keys page
- Environment variable instructions for production use

**Content:**
```markdown
<Note>
**Demo Server by Default**

The code below uses the demo server at `demo.honcho.dev`. Your data is temporary and may be cleared at any time.

**To use your own API key:**
1. Sign up at [app.honcho.dev](https://app.honcho.dev)
2. Generate an API key at [app.honcho.dev/api-keys](https://app.honcho.dev/api-keys)
3. Set it in your environment:
```

### 3. SDK Reference API Key Instructions
**Location:** `/v2/documentation/reference/sdk.mdx` - Lines 133-139

**Added:**
- Info box about default environment
- Link to API keys page
- Production configuration reminder

## Files Changed Summary

| File | Changes |
|------|---------|
| `/v2/documentation/introduction/quickstart.mdx` | Fixed `new Honcho()` → `new Honcho({})`, added demo server warning |
| `/v2/guides/streaming-response.mdx` | Fixed 2 instances of `new Honcho()` → `new Honcho({})` |
| `/v2/guides/dialectic-endpoint.mdx` | Fixed `new Honcho()` → `new Honcho({})` |
| `/v2/documentation/reference/sdk.mdx` | Added info about default environment and API key setup |

## Why These Changes Matter

1. **Prevents TypeScript Errors:** The SDK requires an options object, even if empty
2. **Clear Default Behavior:** Users now understand data goes to demo server by default
3. **Easy Production Path:** Direct links to get API keys for production use
4. **Focused Documentation:** Warnings only in quickstart and SDK reference, not in cookbooks/guides

## Notes
- Cookbooks and other guides left without disclaimers as requested
- Only v2 documentation was modified
- v1 documentation was not touched

---

## Additional Change: TypeScript Async Operations for Peers and Sessions

### Date: 2025-08-26
### Change: Made peer and session creation require `await` in TypeScript

## Changes Made

### Updated TypeScript Examples to Use Await

**Files Modified:**
- `/v2/documentation/introduction/quickstart.mdx` - Added await for peer and session creation
- `/v2/documentation/reference/sdk.mdx` - Updated all TypeScript examples to use await
- `/v2/guides/get-context.mdx` - Added await for peer/session creation
- `/v2/guides/working-rep.mdx` - Added await for peer/session creation

**Pattern Changed:**
```typescript
// Before (incorrect)
const alice = client.peer("alice")
const session = client.session("session_1")

// After (correct)
const alice = await client.peer("alice")
const session = await client.session("session_1")

// For arrays of peers
const users = await Promise.all(
  Array.from({ length: 5 }, (_, i) => honcho.peer(`user-${i}`))
);
```

## Why This Change is Required

The TypeScript SDK returns `Promise<Peer>` and `Promise<Session>` from the creation methods, not the objects directly. Additionally, methods like `addMessages()` and `addPeers()` are also async. This requires:
1. Using `await` for single peer/session creation
2. Using `Promise.all()` for creating multiple peers/sessions
3. Using `await` for all async methods including `addMessages()` and `addPeers()`
4. Ensuring all TypeScript code examples are async-correct

This aligns with the actual TypeScript SDK behavior and prevents runtime errors.

### Final Addition: addMessages await
- Added missing `await` to `session.addMessages()` in quickstart.mdx line 163