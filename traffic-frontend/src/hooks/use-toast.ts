"use client"

import { useState } from 'react'

type ToastType = 'default' | 'destructive'

interface Toast {
  title: string
  description?: string
  variant?: ToastType
}

export function useToast() {
  const [toasts, setToasts] = useState<Toast[]>([])

  const toast = ({ title, description, variant = 'default' }: Toast) => {
    console.log(`Toast (${variant}): ${title}`, description ? `- ${description}` : '')
    
    // You can implement a proper toast system here
    // For now, we'll just log to console and optionally show an alert
    if (variant === 'destructive') {
      alert(`Error: ${title}${description ? `\n${description}` : ''}`)
    } else {
      // For success messages, you could show a temporary notification
      console.log(`✓ ${title}${description ? `: ${description}` : ''}`)
    }
  }

  return { toast, toasts }
}