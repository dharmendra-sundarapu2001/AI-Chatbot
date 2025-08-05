import React from 'react';

/**
 * TicTacToeIcon Component - Renders a grid icon for the Tic-Tac-Toe game
 * @param {Object} props - Component props
 * @param {boolean} props.active - Whether the icon is active/selected
 * @returns {JSX.Element} The icon component
 */
const TicTacToeIcon = ({ active }) => {
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
      className="feather feather-grid"
    >
      <rect x="3" y="3" width="7" height="7"></rect>
      <rect x="14" y="3" width="7" height="7"></rect>
      <rect x="14" y="14" width="7" height="7"></rect>
      <rect x="3" y="14" width="7" height="7"></rect>
    </svg>
  );
};

export default TicTacToeIcon;
