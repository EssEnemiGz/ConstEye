import React from 'react'
import { Link } from 'react-router-dom'
import ButtonBlack from './ui/button-black.tsx'

function Header() {
  return (
    <>
      <header className='sticky flex items-center p-2 px-8 w-full justify-between bg-white bg-opacity-30 backdrop-blur-md border-b border-gray-200'>
        <div className='flex'>
          <Link to="/" className="flex items-center gap-1">
            <img src={"/public/constEye.jpg"} alt="DataMindAi Logo" className='h-8 w-8 sm:w-12 sm:h-12 rounded-lg' />
            <span className="text-lg sm:text-2xl font-bold text-gray-900">ConstEye</span>
          </Link>
        </div>
        <div className='space-x-5'>
          <Link to="/" className='text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors'>Home</Link>
          <Link to="https://github.com/EssEnemiGz/ConstEye/blob/main/docs/exo-classifier-pytorch.md" className='text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors'>Documentation</Link>
          <Link to="https://github.com/EssEnemiGz/ConstEye/blob/main/docs/project-history.md" className='text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors'>About us</Link>
        </div>
        <div className='flex space-x-4'>
          <Link to="https://github.com/EssEnemiGz/ConstEye">
            <ButtonBlack>Github Code</ButtonBlack>
          </Link>
        </div>
      </header>
    </>
  )
}

export default Header
