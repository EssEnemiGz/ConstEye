import React from 'react'

interface FlexibleText {
  children: React.ReactNode;
  className?: string;
}

const Button: React.FC<FlexibleText> = ({ children, className }) => {
  const classes: string = `hover:bg-gray-100 transition-colors rounded-md py-2 px-4 text-base font-medium border border-gray-200 ${className}`
  return (
    <>
      <span className={classes}>
        {children}
      </span>
    </>
  )
}

export default Button
