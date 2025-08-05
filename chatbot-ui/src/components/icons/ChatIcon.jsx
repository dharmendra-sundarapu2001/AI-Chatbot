import React from 'react';

/**
 * ChatIcon Component - Renders a message circle icon for the chat interface
 * @param {Object} props - Component props
 * @param {boolean} props.active - Whether the icon is active/selected
 * @returns {JSX.Element} The icon component
 */
const ChatIcon = ({ active }) => {
  return (
    <svg 
      xmlns="http://www.w3.org/2000/svg" 
      width="24" 
      height="24" 
      viewBox="0 0 24 24" 
      fill="none" 
      stroke={active ? "#2563EB" : "currentColor"} 
      strokeWidth="2" 
      strokeLinecap="round" 
      strokeLinejoin="round" 
      className="feather feather-message-circle"
    >
      <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path>
    </svg>
  );
};

export default ChatIcon;
