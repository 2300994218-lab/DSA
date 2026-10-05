import type React from 'react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { CheckCircle2, Play, RefreshCw, ToggleLeft, ToggleRight } from 'lucide-react';
import { intelligenceApi } from '../../api/intelligence';
import { getParsedApiError, type ParsedApiError } from '../../api/error';
import { useUiLanguage } from '../../contexts/UiLanguageContext';
import { Badge, Button } from '../common';
import { SettingsAlert } from './SettingsAlert';
import { SettingsSectionCard } from './SettingsSectionCard';
import type { IntelligenceFetchResult, IntelligenceSource, IntelligenceSourceTemplate } from '../../types/intelligence';

const REQUIRED_TEMPLATE_IDS = new Set([
  'chinanews-finance-rss',
  'chinanews-scroll-rss',
  'akshare-cls-telegraph',
  'akshare-eastmoney-fastnews',
  'akshare-ths-fastnews',
  'newsnow-jin10',
]);

const NewsSourcesPanel: React.FC = () => {
  const { language } = useUiLanguage();
  const en = language === 'en';
  const [sources, setSources] = useState<IntelligenceSource[]>([]);
  const [templates, setTemplates] = useState<IntelligenceSourceTemplate[]>([]);
  const [loading, setLoading] = useState(true);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [error, setError] = useState<ParsedApiError | null>(null);
  const [notice, setNotice] = useState('');
  const [chainResult, setChainResult] = useState<IntelligenceFetchResult | null>(null);
  const [sourceResults, setSourceResults] = useState<Record<number, IntelligenceFetchResult>>({});

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [nextSources, nextTemplates] = await Promise.all([
        intelligenceApi.listSources(),
        intelligenceApi.listTemplates(),
      ]);
      setSources(nextSources);
      setTemplates(nextTemplates);
    } catch (err: unknown) {
      setError(getParsedApiError(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const sourceByName = useMemo(() => new Map(sources.map((source) => [source.name, source])), [sources]);

  const addTemplate = async (template: IntelligenceSourceTemplate) => {
    const existing = sourceByName.get(template.name);
    setBusyKey(`template:${template.templateId}`);
    setError(null);
    setNotice('');
    try {
      if (existing) {
        const updated = existing.enabled ? existing : await intelligenceApi.setEnabled(existing.id, true);
        setSources((current) => current.map((item) => item.id === updated.id ? updated : item));
        setNotice(en ? `${template.name} is enabled.` : `${template.name} 已启用。`);
      } else {
        const created = await intelligenceApi.createFromTemplate(template.templateId);
        setSources((current) => [created, ...current]);
        setNotice(en ? `${template.name} has been added.` : `${template.name} 已添加。`);
      }
    } catch (err: unknown) {
      setError(getParsedApiError(err));
    } finally {
      setBusyKey(null);
    }
  };

  const toggleSource = async (source: IntelligenceSource) => {
    setBusyKey(`source:${source.id}`);
    setError(null);
    try {
      const updated = await intelligenceApi.setEnabled(source.id, !source.enabled);
      setSources((current) => current.map((item) => item.id === updated.id ? updated : item));
    } catch (err: unknown) {
      setError(getParsedApiError(err));
    } finally {
      setBusyKey(null);
    }
  };

  const testSource = async (source: IntelligenceSource) => {
    setBusyKey(`test:${source.id}`);
    setError(null);
    try {
      const result = await intelligenceApi.testSource(source.id);
      setSourceResults((current) => ({ ...current, [source.id]: result }));
    } catch (err: unknown) {
      setError(getParsedApiError(err));
    } finally {
      setBusyKey(null);
    }
  };

  const testChain = async () => {
    setBusyKey('chain');
    setError(null);
    try {
      setChainResult(await intelligenceApi.testChain());
    } catch (err: unknown) {
      setError(getParsedApiError(err));
    } finally {
      setBusyKey(null);
    }
  };

  if (loading) {
    return <p className="text-sm text-muted-text">{en ? 'Loading news sources…' : '正在加载新闻来源…'}</p>;
  }

  return (
    <div className="space-y-4">
      {error ? <SettingsAlert title={en ? 'News source operation failed' : '新闻来源操作失败'} message={error.message} /> : null}
      {notice ? <SettingsAlert title={en ? 'Done' : '已完成'} message={notice} variant="success" /> : null}

      <SettingsSectionCard
        title={en ? 'AI news search chain' : 'AI 新闻搜索链路'}
        description={en
          ? 'Enable a source here and it becomes available to AI Q&A, single-stock analysis, and market analysis. Tavily and Brave keep their existing priority.'
          : '在这里启用来源后，它会进入 AI 问股、个股分析和大盘分析。Tavily、Brave 保持原有优先级。'}
        actions={(
          <div className="flex flex-wrap gap-2">
            <Button type="button" variant="settings-secondary" size="sm" onClick={() => void load()} disabled={busyKey !== null}>
              <RefreshCw className="h-4 w-4" aria-hidden="true" />{en ? 'Refresh' : '刷新'}
            </Button>
            <Button type="button" variant="settings-primary" size="sm" onClick={() => void testChain()} disabled={busyKey !== null || sources.every((source) => !source.enabled)} isLoading={busyKey === 'chain'}>
              <Play className="h-4 w-4" aria-hidden="true" />{en ? 'Test enabled chain' : '测试完整链路'}
            </Button>
          </div>
        )}
      >
        <p className="text-xs leading-6 text-muted-text">
          {en ? 'The chain test uses dry-run mode: it fetches each enabled source without writing news items.' : '完整链路测试使用 dry-run 模式：会逐个请求已启用来源，但不会写入资讯。'}
        </p>
        {chainResult ? (
          <div className="rounded-lg border settings-border bg-background/30 p-3 text-sm">
            <p className="font-medium text-foreground">
              {chainResult.ok ? (en ? 'Chain completed' : '完整链路完成') : (en ? 'Chain completed with errors' : '完整链路有失败来源')}
              {' · '}{en ? 'sources' : '来源'} {chainResult.sourceCount ?? 0}{' · '}{en ? 'fetched' : '抓取'} {chainResult.results?.reduce((sum, item) => sum + (item.fetchedCount ?? 0), 0) ?? 0}
            </p>
            <div className="mt-2 space-y-1 text-xs text-muted-text">
              {(chainResult.results || []).map((item) => <p key={item.sourceId}>{item.ok ? '✓' : '✗'} {item.sourceId} · {item.fetchedCount ?? 0} {item.error || ''}</p>)}
            </div>
          </div>
        ) : null}
      </SettingsSectionCard>

      <SettingsSectionCard title={en ? 'Recommended free sources' : '推荐的免费来源'} description={en ? 'Add a built-in template, then test it before keeping it enabled.' : '添加内置模板后可先单源测试，再决定是否保持启用。'}>
        <div className="grid gap-3 md:grid-cols-2">
          {templates.map((template) => {
            const existing = sourceByName.get(template.name);
            const isRequired = REQUIRED_TEMPLATE_IDS.has(template.templateId);
            return (
              <div key={template.templateId} className="rounded-lg border settings-border bg-background/30 p-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-foreground">{template.name}</p>
                    <p className="mt-1 text-xs text-muted-text">{template.description}</p>
                  </div>
                  <Badge variant={isRequired ? 'info' : 'default'} size="sm">{isRequired ? (en ? 'Required' : '指定来源') : template.market.toUpperCase()}</Badge>
                </div>
                <div className="mt-3 flex items-center justify-between gap-2">
                  <span className="text-[11px] text-muted-text">{template.sourceType} · {template.market}</span>
                  <Button type="button" variant="settings-secondary" size="xsm" onClick={() => void addTemplate(template)} disabled={busyKey !== null} isLoading={busyKey === `template:${template.templateId}`}>
                    {existing?.enabled ? (en ? 'Enabled' : '已启用') : existing ? (en ? 'Enable' : '启用') : (en ? 'Add & enable' : '添加并启用')}
                  </Button>
                </div>
              </div>
            );
          })}
        </div>
      </SettingsSectionCard>

      <SettingsSectionCard title={en ? 'Configured sources' : '已配置来源'} description={en ? 'Only enabled rows are included in the AI search provider chain.' : '只有已启用的来源会进入 AI 搜索提供方链路。'}>
        {sources.length === 0 ? <p className="text-sm text-muted-text">{en ? 'No source configured yet.' : '还没有配置来源。'}</p> : (
          <div className="space-y-3">
            {sources.map((source) => {
              const result = sourceResults[source.id];
              return (
                <div key={source.id} className="rounded-lg border settings-border bg-background/30 p-3">
                  <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2"><p className="text-sm font-medium text-foreground">{source.name}</p><Badge variant={source.enabled ? 'success' : 'default'} size="sm">{source.enabled ? (en ? 'Enabled' : '已启用') : (en ? 'Disabled' : '未启用')}</Badge><Badge variant="default" size="sm">{source.sourceType}</Badge></div>
                      <p className="mt-1 break-all text-xs text-muted-text">{source.url}</p>
                      {source.lastError ? <p className="mt-1 text-xs text-red-600 dark:text-red-300">{source.lastError}</p> : null}
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <Button type="button" variant="settings-secondary" size="xsm" onClick={() => void toggleSource(source)} disabled={busyKey !== null} isLoading={busyKey === `source:${source.id}`}>
                        {source.enabled ? <ToggleRight className="h-4 w-4" aria-hidden="true" /> : <ToggleLeft className="h-4 w-4" aria-hidden="true" />}{source.enabled ? (en ? 'Disable' : '停用') : (en ? 'Enable' : '启用')}
                      </Button>
                      <Button type="button" variant="settings-secondary" size="xsm" onClick={() => void testSource(source)} disabled={busyKey !== null || !source.enabled} isLoading={busyKey === `test:${source.id}`}>
                        <Play className="h-4 w-4" aria-hidden="true" />{en ? 'Test source' : '单源测试'}
                      </Button>
                    </div>
                  </div>
                  {result ? <p className="mt-2 text-xs text-muted-text"><CheckCircle2 className="mr-1 inline h-3.5 w-3.5 text-emerald-500" aria-hidden="true" />{en ? 'Fetched' : '抓取'} {result.fetchedCount ?? 0} {en ? 'items' : '条'}{result.error ? ` · ${result.error}` : ''}</p> : null}
                </div>
              );
            })}
          </div>
        )}
      </SettingsSectionCard>
    </div>
  );
};

export default NewsSourcesPanel;
