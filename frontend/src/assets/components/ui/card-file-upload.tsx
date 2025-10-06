
import React from "react"
import clsx from "clsx"

export const Card: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ className, children, ...props }) => (
  <div
    className={clsx(
      "bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex flex-col gap-4",
      className
    )}
    {...props}
  >
    {children}
  </div>
)

export const CardHeader: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ className, children, ...props }) => (
  <div className={clsx("border-b pb-4 mb-2", className)} {...props}>
    {children}
  </div>
)

export const CardTitle: React.FC<React.HTMLAttributes<HTMLHeadingElement>> = ({ className, children, ...props }) => (
  <h2 className={clsx("text-lg font-semibold", className)} {...props}>
    {children}
  </h2>
)

export const CardDescription: React.FC<React.HTMLAttributes<HTMLParagraphElement>> = ({ className, children, ...props }) => (
  <p className={clsx("text-gray-500 text-sm", className)} {...props}>
    {children}
  </p>
)

export const CardContent: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ className, children, ...props }) => (
  <div className={clsx("flex-1", className)} {...props}>
    {children}
  </div>
)

export const CardFooter: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ className, children, ...props }) => (
  <div className={clsx("border-t pt-4 mt-2 flex items-center justify-end", className)} {...props}>
    {children}
  </div>
)

