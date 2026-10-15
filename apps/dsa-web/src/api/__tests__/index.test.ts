// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import apiClient from '../index';
import { parseApiError } from '../error';

function setLocation(pathname: string, search = ''): { assignMock: ReturnType<typeof vi.fn> } {
  const assignMock = vi.fn();
  Object.defineProperty(window, 'location', {
    configurable: true,
    writable: true,
    value: { pathname, search, assign: assignMock },
  });
  return { assignMock };
}

function rejectWith(error: Record<string, unknown>): void {
  apiClient.defaults.adapter = async (config) => {
    const err = new Error((error.message as string) ?? 'request failed');
    Object.assign(err, error, { config });
    throw err;
  };
}

afterEach(() => {
  Object.defineProperty(window, 'location', {
    configurable: true,
    writable: true,
    value: { pathname: '/', search: '', assign: vi.fn() },
  });
});

describe('apiClient 401 interceptor', () => {
  it('redirects to login with the current path when 401 on a normal page', async () => {
    const { assignMock } = setLocation('/portfolio', '?tab=1');
    rejectWith({ response: { status: 401, data: { detail: 'Unauthorized' } }, message: 'Request failed with status code 401' });

    await expect(apiClient.get('/api/v1/portfolio')).rejects.toBeTruthy();

    expect(assignMock).toHaveBeenCalledTimes(1);
    expect(assignMock).toHaveBeenCalledWith('/login?redirect=%2Fportfolio%3Ftab%3D1');
  });

  it('does not redirect when 401 happens on the login page itself', async () => {
    const { assignMock } = setLocation('/login', '');
    rejectWith({ response: { status: 401, data: { detail: 'Bad credentials' } }, message: 'Request failed with status code 401' });

    await expect(apiClient.get('/api/v1/auth/login')).rejects.toBeTruthy();

    expect(assignMock).not.toHaveBeenCalled();
  });

  it('does not redirect on non-401 failures', async () => {
    const { assignMock } = setLocation('/analysis');
    rejectWith({ response: { status: 500, data: { detail: 'boom' } }, message: 'Request failed with status code 500' });

    await expect(apiClient.get('/api/v1/analysis')).rejects.toBeTruthy();

    expect(assignMock).not.toHaveBeenCalled();
  });

  it('attaches a parsed error with payload text to the rejected error', async () => {
    setLocation('/analysis');
    rejectWith({ response: { status: 500, data: { detail: 'boom' } }, message: 'Request failed with status code 500' });

    const error = await apiClient.get('/api/v1/analysis').catch((err) => err);

    expect(error.parsedError).toBeDefined();
    expect(error.parsedError.category).toBe('http_error');
    expect(error.parsedError.message).toBe('boom');
    expect(error.message).toContain('boom');
  });
});

describe('parseApiError', () => {
  it('classifies client-side timeouts', () => {
    const parsed = parseApiError({ code: 'ECONNABORTED', message: 'timeout of 30000ms exceeded' });
    expect(parsed.category).toBe('upstream_timeout');
  });

  it('classifies local connection failures without a response', () => {
    const parsed = parseApiError({ message: 'Network Error' });
    expect(parsed.category).toBe('local_connection_failed');
  });

  it('classifies missing LLM configuration from the payload text', () => {
    const parsed = parseApiError({
      response: { status: 500, data: { detail: 'No LLM configured' } },
    });
    expect(parsed.category).toBe('llm_not_configured');
  });

  it('classifies portfolio oversell by error code', () => {
    const parsed = parseApiError({
      response: { status: 400, data: { detail: { error: 'portfolio_oversell' } } },
    });
    expect(parsed.category).toBe('portfolio_oversell');
  });

  it('falls back to http_error with the payload text for plain failures', () => {
    // 502/503 会被归类为 upstream_network（服务端无法访问外部依赖）；
    // 普通 4xx/5xx 才走 http_error 兜底并展示 payload 文本。
    const parsed = parseApiError({ response: { status: 400, data: { detail: 'bad request detail' } } });
    expect(parsed.category).toBe('http_error');
    expect(parsed.message).toBe('bad request detail');
  });
});
