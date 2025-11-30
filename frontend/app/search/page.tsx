'use client';

import { useState } from 'react';
import { api, type SearchResult } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Search as SearchIcon, AlertCircle, Loader2 } from 'lucide-react';
import { toast } from '@/lib/toast';

export default function SearchPage() {
  const [master, setMaster] = useState('');
  const [query, setQuery] = useState('{\n  "fname": "",\n  "lname": ""\n}');
  const [threshold, setThreshold] = useState('0.85');
  const [maxResults, setMaxResults] = useState('10');
  const [results, setResults] = useState<SearchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    setResults(null);

    try {
      let queryObj;
      try {
        queryObj = JSON.parse(query);
      } catch (err) {
        throw new Error('Invalid JSON query');
      }

      const thresholdNum = parseFloat(threshold);
      if (isNaN(thresholdNum) || thresholdNum < 0 || thresholdNum > 1) {
        throw new Error('Threshold must be a number between 0 and 1');
      }

      const maxResultsNum = parseInt(maxResults);
      if (isNaN(maxResultsNum) || maxResultsNum < 1) {
        throw new Error('Max results must be a positive integer');
      }

      const result = await api.search({
        master: master.trim(),
        query: queryObj,
        threshold: thresholdNum,
        max_results: maxResultsNum,
      });

      setResults(result);
      toast(`Found ${result.count} match${result.count !== 1 ? 'es' : ''}`, 'success');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Search failed';
      setError(message);
      toast(message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.9) return 'bg-green-500';
    if (score >= 0.7) return 'bg-blue-500';
    if (score >= 0.5) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Search</h1>
        <p className="text-muted-foreground mt-1">Search for matching records in a master dataset</p>
      </div>

      <form onSubmit={handleSearch}>
        <Card>
          <CardHeader>
            <CardTitle>Search Parameters</CardTitle>
            <CardDescription>Configure your search query</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="master">Master Dataset *</Label>
              <Input
                id="master"
                value={master}
                onChange={(e) => setMaster(e.target.value)}
                placeholder="data/master.csv or s3://bucket/file.csv or mysql://table"
                required
              />
              <p className="text-sm text-muted-foreground">
                Path to CSV file, S3 URL, or MySQL table name
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="query">Query Record (JSON) *</Label>
              <Textarea
                id="query"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="font-mono text-sm"
                rows={6}
                required
              />
              <p className="text-sm text-muted-foreground">
                JSON object with fields to search for (e.g., {"{"}"fname": "John", "lname": "Smith"{"}"})
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="threshold">Threshold</Label>
                <Input
                  id="threshold"
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  value={threshold}
                  onChange={(e) => setThreshold(e.target.value)}
                  required
                />
                <p className="text-sm text-muted-foreground">
                  Minimum match score (0-1)
                </p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="maxResults">Max Results</Label>
                <Input
                  id="maxResults"
                  type="number"
                  min="1"
                  value={maxResults}
                  onChange={(e) => setMaxResults(e.target.value)}
                  required
                />
                <p className="text-sm text-muted-foreground">
                  Maximum number of results
                </p>
              </div>
            </div>

            <Button type="submit" disabled={loading} className="w-full">
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Searching...
                </>
              ) : (
                <>
                  <SearchIcon className="h-4 w-4 mr-2" />
                  Search
                </>
              )}
            </Button>
          </CardContent>
        </Card>
      </form>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {results && (
        <Card>
          <CardHeader>
            <CardTitle>Search Results</CardTitle>
            <CardDescription>
              Found {results.count} match{results.count !== 1 ? 'es' : ''} in {results.master_dataset}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {results.count === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                No matches found
              </div>
            ) : (
              <div className="space-y-4">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Score</TableHead>
                      <TableHead>Record</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {results.matches.map((match, index) => (
                      <TableRow key={index}>
                        <TableCell>
                          <Badge className={getScoreColor(match.score)}>
                            {(match.score * 100).toFixed(1)}%
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <pre className="text-sm bg-muted p-2 rounded">
                            {JSON.stringify(match.record, null, 2)}
                          </pre>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

