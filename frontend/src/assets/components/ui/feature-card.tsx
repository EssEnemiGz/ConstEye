import React from 'react';

interface FlexibleContent {
  icon: React.ReactNode;
  title: string;
  description: string;

}

const FeatureCard: React.FC<FlexibleContent> = ({ icon, title, description }) => {
  return (
    <>
      <div className="rounded-lg bg-card text-card-foreground border-0 shadow-lg hover:shadow-xl transition-shadow">
        <div className='flex flex-col space-y-1.5 p-6'>
          {icon}
          <h3 className="text-2xl font-semibold leading-none tracking-tight">{title}</h3>
          <p className="text-sm text-muted-foreground font-mediumtext-xl text-gray-600">{description}</p>
        </div>
      </div>
    </>
  )
}

export default FeatureCard
