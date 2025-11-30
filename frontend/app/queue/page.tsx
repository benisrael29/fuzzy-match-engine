'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { api, type QueueJob } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { ArrowLeft, RefreshCw, AlertCircle, Loader2, Clock, Play } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { toast } from '@/lib/toast';

export default function QueuePage() {
  const [queue, setQueue] = useState<QueueJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [polling, setPolling] = useState(false);

  const fetchQueue = async () => {
    try {
      setError(null);
      const data = await api.getQueue();
      setQueue(data);
      
      // Auto-poll if there are queued or running jobs
      const hasActiveJobs = data.some(job => 
        job.status === 'queued' || job.status === 'running' || job.status === 'cancelling'
      );
      setPolling(hasActiveJobs);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load queue');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchQueue();
  }, []);

  useEffect(() => {
    if (polling) {
      const interval = setInterval(fetchQueue, 2000);
      return () => clearInterval(interval);
    }
  }, [polling]);

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'queued':
        return (
          <Badge variant="default" className="bg-yellow-500">
            <Clock className="h-3 w-3 mr-1" />
            Queued
          </Badge>
        );
      case 'running':
        return (
          <Badge variant="default" className="bg-blue-500">
            <Loader2 className="h-3 w-3 mr-1 animate-spin" />
            Running
          </Badge>
        );
      case 'cancelling':
        return (
          <Badge variant="default" className="bg-orange-500">
            <Loader2 className="h-3 w-3 mr-1 animate-spin" />
            Cancelling
          </Badge>
        );
      default:
        return <Badge variant="secondary">{status}</Badge>;
    }
  };

  const getPriorityBadge = (priority: string) => {
    const colors = {
      high: 'bg-red-500',
      medium: 'bg-yellow-500',
      low: 'bg-gray-500'
    };
    return (
      <Badge variant="outline" className={colors[priority as keyof typeof colors] || ''}>
        {priority.charAt(0).toUpperCase() + priority.slice(1)}
      </Badge>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="flex flex-col items-center gap-2">
          <Loader2 className="h-6 w-6 animate-spin text-primary" />
          <div className="text-muted-foreground">Loading queue...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="sm" asChild>
            <Link href="/">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Link>
          </Button>
          <div>
            <h1 className="text-3xl font-bold">Job Queue</h1>
            <p className="text-muted-foreground mt-1">View and manage queued and running jobs</p>
          </div>
        </div>
        <Button
          variant="outline"
          onClick={fetchQueue}
          disabled={loading}
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {queue.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Clock className="h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-muted-foreground mb-4">Queue is empty</p>
            <Button asChild>
              <Link href="/">
                <Play className="h-4 w-4 mr-2" />
                Queue a Job
              </Link>
            </Button>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>Queue Status</CardTitle>
            <CardDescription>
              {queue.filter(j => j.status === 'queued').length} queued, {queue.filter(j => j.status === 'running').length} running
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Position</TableHead>
                  <TableHead>Job Name</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Priority</TableHead>
                  <TableHead>Queued At</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {queue.map((job) => (
                  <TableRow key={job.job_name}>
                    <TableCell>
                      {job.queue_position ? (
                        <span className="font-mono">#{job.queue_position}</span>
                      ) : (
                        <span className="text-muted-foreground">-</span>
                      )}
                    </TableCell>
                    <TableCell className="font-medium">
                      <Link
                        href={`/jobs/${encodeURIComponent(job.job_name)}`}
                        className="text-primary hover:underline"
                      >
                        {job.job_name}
                      </Link>
                    </TableCell>
                    <TableCell>
                      {getStatusBadge(job.status)}
                    </TableCell>
                    <TableCell>
                      {getPriorityBadge(job.priority)}
                    </TableCell>
                    <TableCell>
                      {formatDistanceToNow(new Date(job.queued_at), { addSuffix: true })}
                    </TableCell>
                    <TableCell className="text-right">
                      <Button
                        variant="outline"
                        size="sm"
                        asChild
                      >
                        <Link href={`/jobs/${encodeURIComponent(job.job_name)}`}>
                          View
                        </Link>
                      </Button>
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

