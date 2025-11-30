import Link from 'next/link';
import Image from 'next/image';
import { Button } from '@/components/ui/button';
import { Search, List } from 'lucide-react';
import { ConnectionStatus } from '@/components/connection-status';

export function Header() {
  return (
    <header className="border-b bg-card">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <Link href="/" className="flex items-center gap-3">
            <Image
              src="/logo.svg"
              alt="Fuzzy Matching Engine"
              width={32}
              height={32}
              className="h-8 w-8"
            />
            <span className="text-xl font-semibold">Fuzzy Matching Engine</span>
          </Link>
          <div className="flex items-center gap-4">
            <ConnectionStatus />
            <nav className="flex items-center gap-2">
              <Button variant="ghost" asChild>
                <Link href="/">
                  <List className="h-4 w-4 mr-2" />
                  Jobs
                </Link>
              </Button>
              <Button variant="ghost" asChild>
                <Link href="/search">
                  <Search className="h-4 w-4 mr-2" />
                  Search
                </Link>
              </Button>
            </nav>
          </div>
        </div>
      </div>
    </header>
  );
}

