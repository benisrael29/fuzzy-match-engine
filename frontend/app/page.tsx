'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { api, type JobList } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Plus, Play, Edit, Trash2, AlertCircle, Loader2 } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { toast } from '@/lib/toast';

export default function JobsPage() {
  const [jobs, setJobs] = useState<JobList[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  const fetchJobs = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.listJobs();
      setJobs(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load jobs');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchJobs();
  }, []);

  const handleDelete = async (name: string) => {
    if (!confirm(`Are you sure you want to delete job "${name}"?`)) {
      return;
    }

    try {
      setDeleting(name);
      await api.deleteJob(name);
      toast(`Job "${name}" deleted successfully`, 'success');
      await fetchJobs();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete job';
      toast(message, 'error');
    } finally {
      setDeleting(null);
    }
  };

  const handleRun = async (name: string) => {
    try {
      const result = await api.runJob(name, 'medium');
      toast(`Job "${name}" queued successfully${result.queue_position ? ` (position: ${result.queue_position})` : ''}`, 'success');
      window.location.href = `/jobs/${encodeURIComponent(name)}`;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to queue job';
      toast(message, 'error');
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="flex flex-col items-center gap-2">
          <Loader2 className="h-6 w-6 animate-spin text-primary" />
          <div className="text-muted-foreground">Loading jobs...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Jobs</h1>
          <p className="text-muted-foreground mt-1">Manage your fuzzy matching jobs</p>
        </div>
        <Button asChild>
          <Link href="/jobs/new">
            <Plus className="h-4 w-4 mr-2" />
            Create Job
          </Link>
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {jobs.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <p className="text-muted-foreground mb-4">No jobs found</p>
            <Button asChild>
              <Link href="/jobs/new">
                <Plus className="h-4 w-4 mr-2" />
                Create Your First Job
              </Link>
            </Button>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>All Jobs</CardTitle>
            <CardDescription>View and manage all your matching jobs</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Description</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead>Modified</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {jobs.map((job) => (
                  <TableRow key={job.name}>
                    <TableCell className="font-medium">
                      <Link
                        href={`/jobs/${encodeURIComponent(job.name)}`}
                        className="text-primary hover:underline"
                      >
                        {job.name}
                      </Link>
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {job.description || '-'}
                    </TableCell>
                    <TableCell>
                      {formatDistanceToNow(new Date(job.created), { addSuffix: true })}
                    </TableCell>
                    <TableCell>
                      {formatDistanceToNow(new Date(job.modified), { addSuffix: true })}
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex items-center justify-end gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleRun(job.name)}
                        >
                          <Play className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          asChild
                        >
                          <Link href={`/jobs/${encodeURIComponent(job.name)}/edit`}>
                            <Edit className="h-4 w-4" />
                          </Link>
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleDelete(job.name)}
                          disabled={deleting === job.name}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
