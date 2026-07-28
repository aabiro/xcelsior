import { AsyncLocalStorage } from "node:async_hooks";

export interface ApiCallRecord {
  route: string;
  method: string;
  status: number;
  problemType?: string;
}

const storage = new AsyncLocalStorage<ApiCallRecord[]>();

export function withApiCapture<T>(calls: ApiCallRecord[], fn: () => Promise<T>): Promise<T> {
  return storage.run(calls, fn);
}

export function captureApiCall(record: ApiCallRecord): void {
  storage.getStore()?.push(record);
}
