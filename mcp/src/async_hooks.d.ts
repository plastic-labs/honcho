// http.ts scopes token rejection per request with AsyncLocalStorage. The
// project type-checks against the Workers types only, so declare what it uses.
declare module "node:async_hooks" {
  export class AsyncLocalStorage<T> {
    run<R>(store: T, callback: () => R): R;
    getStore(): T | undefined;
  }
}
