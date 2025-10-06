import React from 'react'

interface FlexibleText {
  children: React.ReactNode;
  className?: string;
}

const ButtonBlack: React.FC<FlexibleText> = ({ children, className }) => {
  const classes: string = `hover:bg-gray-800 bg-black py-2 px-4 transition-colors rounded-md text-white ${className}`
  return (
    <>
      <span className={classes}>
        {children}
      </span>
    </>
  )
}

export default ButtonBlack
