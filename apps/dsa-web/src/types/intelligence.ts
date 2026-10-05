export interface IntelligenceSource {
  id: number;
  name: string;
  sourceType: string;
  url: string;
  enabled: boolean;
  scopeType: string;
  scopeValue?: string | null;
  market: string;
  description?: string | null;
  lastStatus?: string | null;
  lastError?: string | null;
  lastFetchedAt?: string | null;
  createdAt?: string | null;
  updatedAt?: string | null;
}

export interface IntelligenceSourceTemplate {
  templateId: string;
  name: string;
  sourceType: string;
  url: string;
  scopeType: string;
  scopeValue?: string | null;
  market: string;
  description?: string | null;
}

export interface IntelligenceFetchResult {
  ok: boolean;
  sourceId?: number;
  sourceCount?: number;
  fetchedCount?: number;
  savedCount?: number;
  dryRun?: boolean;
  sampleItems?: Array<{ title: string; summary?: string | null; url: string; publishedAt?: string | null }>;
  results?: Array<IntelligenceFetchResult & { error?: string }>;
  error?: string;
}
