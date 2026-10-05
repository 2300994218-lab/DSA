import apiClient from './index';
import { toCamelCase } from './utils';
import type { IntelligenceFetchResult, IntelligenceSource, IntelligenceSourceTemplate } from '../types/intelligence';

interface SourceListResponse {
  items: IntelligenceSource[];
  total: number;
}

interface TemplateListResponse {
  items: IntelligenceSourceTemplate[];
  total: number;
}

export const intelligenceApi = {
  async listSources(): Promise<IntelligenceSource[]> {
    const response = await apiClient.get<SourceListResponse>('/api/v1/intelligence/sources', { params: { page_size: 100 } });
    return toCamelCase<SourceListResponse>(response.data).items || [];
  },

  async listTemplates(): Promise<IntelligenceSourceTemplate[]> {
    const response = await apiClient.get<TemplateListResponse>('/api/v1/intelligence/sources/templates');
    return toCamelCase<TemplateListResponse>(response.data).items || [];
  },

  async createFromTemplate(templateId: string): Promise<IntelligenceSource> {
    const response = await apiClient.post<IntelligenceSource>(`/api/v1/intelligence/sources/templates/${templateId}`, {});
    return toCamelCase<IntelligenceSource>(response.data);
  },

  async setEnabled(sourceId: number, enabled: boolean): Promise<IntelligenceSource> {
    const response = await apiClient.patch<IntelligenceSource>(`/api/v1/intelligence/sources/${sourceId}`, { enabled });
    return toCamelCase<IntelligenceSource>(response.data);
  },

  async testSource(sourceId: number): Promise<IntelligenceFetchResult> {
    const response = await apiClient.post<IntelligenceFetchResult>(`/api/v1/intelligence/sources/${sourceId}/fetch`, null, { params: { dry_run: true } });
    return toCamelCase<IntelligenceFetchResult>(response.data);
  },

  async testChain(): Promise<IntelligenceFetchResult> {
    const response = await apiClient.post<IntelligenceFetchResult>('/api/v1/intelligence/sources/fetch-enabled', null, { params: { dry_run: true } });
    return toCamelCase<IntelligenceFetchResult>(response.data);
  },
};
