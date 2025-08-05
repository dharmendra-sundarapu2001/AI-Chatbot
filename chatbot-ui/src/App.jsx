import React, { useState, useRef, useEffect } from 'react';
import { GoogleOAuthProvider, GoogleLogin } from '@react-oauth/google';
import MessageRenderer from './components/MessageRenderer'; // Assuming MessageRenderer is in a components folder
import TicTacToeGame from './components/TicTacToeGame';
import TicTacToeIcon from './components/icons/TicTacToeIcon';
import ChatIcon from './components/icons/ChatIcon';

// Use Vite's import.meta.env for frontend env variables
// IMPORTANT: Ensure your .env file has VITE_GOOGLE_CLIENT_ID=YOUR_CLIENT_ID
const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID;

function App() {
    // Auth state
    const [isLoggedIn, setIsLoggedIn] = useState(() => localStorage.getItem('isLoggedIn') === 'true');
    const [loggedInEmail, setLoggedInEmail] = useState(() => localStorage.getItem('loggedInEmail') || '');
    const [showSignup, setShowSignup] = useState(false);
    const [authEmail, setAuthEmail] = useState('');
    const [authPassword, setAuthPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false); // State for password visibility
    const [authError, setAuthError] = useState('');
    const [authSuccess, setAuthSuccess] = useState('');
    const [signOutMsg, setSignOutMsg] = useState('');
    const [searchTerm, setSearchTerm] = useState(""); // Add searchTerm state here, early in the component

    // Theme state - default to blue and include dark mode
    const [isDarkMode, setIsDarkMode] = useState(() => localStorage.getItem('isDarkMode') === 'true'); // Initialize dark mode state
    const [theme, setTheme] = useState(() => 'blue'); // Default to 'blue' theme
    
    // View state - for switching between chat and game
    const [activeView, setActiveView] = useState('chat'); // 'chat' or 'game'

    // Chat & Thread state
    const [threads, setThreads] = useState([]);
    const [activeThreadId, setActiveThreadId] = useState(() => {
        const savedThreadId = sessionStorage.getItem('activeThreadId');
        return savedThreadId ? parseInt(savedThreadId, 10) : null;
    });
    const [question, setQuestion] = useState('');
    const [chat, setChat] = useState([]);
    const [isBotTyping, setIsBotTyping] = useState(false); // State for bot typing indicator
    const [isWebSearching, setIsWebSearching] = useState(false); // State for web search indicator
    const [isLoadingChat, setIsLoadingChat] = useState(false); // NEW: State to indicate chat messages are loading
    const [profileOpen, setProfileOpen] = useState(false);

    // Sidebar state - now only controlled by click, no hover
    const [isSidebarOpen, setIsSidebarOpen] = useState(false); // Default to CLOSED on reload

    const [editingThreadId, setEditingThreadId] = useState(null);
    const [editingTitle, setEditingTitle] = useState(''); // Corrected initialization
    const [optionsMenuThreadId, setOptionsMenuThreadId] = useState(null);
    const inputRef = useRef(null);
    const chatEndRef = useRef(null);
    const profileRef = useRef(null); // Ref for profile dropdown
    const chatContainerRef = useRef(null); // Ref for the chat messages container


    // Ref for the sidebar itself, crucial for outside click detection
    const sidebarRef = useRef(null);
    const optionsMenuRef = useRef(null);
    const themeSelectorRef = useRef(null); // Ref for theme selector (not used in provided code, but kept)

    const [selectedModel, setSelectedModel] = useState('chatgpt'); // Default to ChatGPT

    // File Upload state (modified for generic files)
    const [selectedFile, setSelectedFile] = useState(null); // Stores the File object
    const [filePreviewUrl, setFilePreviewUrl] = useState(null); // Stores the URL for preview (e.g., image, PDF icon)
    const [fileType, setFileType] = useState(null); // Stores the type of file (e.g., 'image', 'pdf', 'other')
    const fileInputRef = useRef(null); // Ref for the hidden file input

    // State for scroll button visibility
    const [showScrollToBottom, setShowScrollToBottom] = useState(false);

    // Theme configurations - Only 'blue' theme is kept as requested
    const themes = {
        blue: {
            name: 'Ocean Blue',
            primary: '#4fc3f7',
            primaryHover: '#29b6f6',
            secondary: '#1976d2',
            accent: '#0277bd',
            background: '#000000',
            surface: '#171717',
            surfaceHover: '#262626'
        },
    };

    const currentTheme = themes[theme]; // This will always be the 'blue' theme

    // Generate CSS custom properties for current theme
    const themeStyles = {
        '--color-primary': currentTheme.primary,
        '--color-primary-hover': currentTheme.primaryHover,
        '--color-secondary': currentTheme.secondary,
        '--color-accent': currentTheme.accent,
    };

    // Close dropdowns and sidebar on outside click
    useEffect(() => {
        function handleClickOutside(event) {
            // Close profile dropdown
            if (profileRef.current && !profileRef.current.contains(event.target)) {
                setProfileOpen(false);
            }
            // Close thread options menu
            if (optionsMenuRef.current && !optionsMenuRef.current.contains(event.target)) {
                setOptionsMenuThreadId(null);
            }
            // Close sidebar on outside click ONLY if it's explicitly opened
            // AND the click wasn't on the menu toggle button itself.
            const menuToggleButton = document.getElementById('sidebar-toggle-button');
            if (isSidebarOpen && sidebarRef.current &&
                !sidebarRef.current.contains(event.target) &&
                event.target !== menuToggleButton &&
                !menuToggleButton.contains(event.target)) {
                setIsSidebarOpen(false);
            }
        }

        document.addEventListener("mousedown", handleClickOutside);

        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, [profileRef, optionsMenuRef, isSidebarOpen]);

    // Save theme to localStorage
    useEffect(() => {
        localStorage.setItem('theme', theme);
    }, [theme]);

    // Save dark mode preference to localStorage
    useEffect(() => {
        localStorage.setItem('isDarkMode', isDarkMode.toString());
    }, [isDarkMode]);

    // Fetch threads on login
    useEffect(() => {
        if (isLoggedIn && loggedInEmail) {
            fetch('http://localhost:8000/threads', {
                headers: { "X-User-Email": loggedInEmail }
            })
                .then(res => res.json())
                .then(data => {
                    setThreads(data.map(thread => ({ ...thread, isLoading: false })));
                    if (!activeThreadId) {
                        setChat([]);
                    }
                })
                .catch(error => console.error("Failed to fetch threads:", error));
        }
    }, [isLoggedIn, loggedInEmail, activeThreadId]);

    // Fetch chats when active thread changes
    useEffect(() => {
        if (activeThreadId) {
            // Set loading for the specific active thread
            setThreads(prevThreads =>
                prevThreads.map(thread =>
                    thread.id === activeThreadId ? { ...thread, isLoading: true } : thread
                )
            );
            // Always clear chat and typing indicators immediately on thread switch
            setChat([]); // Clear chat instantly for visual freshness
            setIsBotTyping(false);
            setIsWebSearching(false);
            setIsLoadingChat(true); // NEW: Set chat loading to true before fetching

            fetch(`http://localhost:8000/threads/${activeThreadId}/chats`, {
                headers: { "X-User-Email": loggedInEmail }
            })
                .then(res => res.json())
                .then(data => {
                    setChat(data.map(msg => ({
                        type: msg.sender,
                        text: msg.message,
                        timestamp: msg.timestamp,
                        id: msg.id,
                        image_data_base64: msg.image_data_base64 || null,
                        image_mime_type: msg.image_mime_type || null,
                        video_data_base64: msg.video_data_base64 || null,
                        video_mime_type: msg.video_mime_type || null,
                        filename: msg.filename || null,
                        websearch_info: msg.websearch_info || null,
                    })));
                })
                .catch(error => {
                    console.error("Failed to fetch thread chats:", error);
                    // Optionally display an error message in chat if fetch fails
                    setChat([{ type: 'bot', text: `Error loading chat history: ${error.message}` }]);
                })
                .finally(() => {
                    // Set loading to false for the specific active thread after fetch completes
                    setThreads(prevThreads =>
                        prevThreads.map(thread =>
                            thread.id === activeThreadId ? { ...thread, isLoading: false } : thread
                        )
                    );
                    setIsLoadingChat(false); // NEW: Set chat loading to false after fetch completes
                });
        } else {
            // If activeThreadId is null (new chat), ensure chat is cleared and no loading state
            setChat([]);
            setIsLoadingChat(false); // NEW: No chat loading when starting fresh
        }
    }, [activeThreadId, loggedInEmail]); // Dependency array to re-run on activeThreadId change

    // Scroll to bottom of chat and manage scroll button visibility
    useEffect(() => {
        const chatContainer = chatContainerRef.current;
        if (chatContainer) {
            const handleScroll = () => {
                const { scrollTop, scrollHeight, clientHeight } = chatContainer;
                // Show button if not at the bottom (within a 10px threshold)
                setShowScrollToBottom(scrollHeight - scrollTop > clientHeight + 10);
            };

            // Set up listener
            chatContainer.addEventListener('scroll', handleScroll);
            // Clean up listener
            return () => chatContainer.removeEventListener('scroll', handleScroll);
        }
    }, [chat]); // Re-run when chat messages change to update scroll state

    // Initial scroll to bottom and scroll on new messages
    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [chat, activeThreadId, isLoadingChat]); // Added isLoadingChat to trigger scroll when chat loads

    // Reset textarea height when thread changes
    useEffect(() => {
        const textarea = document.querySelector('textarea[placeholder="Type a message..."]');
        if (textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = ''; // Reset to CSS default
            textarea.value = '';
        }
        setQuestion(''); // Also reset the question state
    }, [activeThreadId]);

    const handleLogin = async () => {
        setAuthError('');
        setAuthSuccess('');
        if (!authEmail || !authPassword) {
            setAuthError('Please enter both email and password.');
            return;
        }
        try {
            const res = await fetch('http://localhost:8000/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email: authEmail, password: authPassword }),
            });
            const data = await res.json();
            if (!res.ok) {
                if (data.detail === "Invalid credentials") {
                    setAuthError("Invalid credentials. Please try again.");
                } else if (data.detail?.toLowerCase().includes("not found")) {
                    setAuthError("No account found. Please sign up to use.");
                } else {
                    setAuthError(data.detail || 'Login failed');
                }
            } else {
                setIsLoggedIn(true);
                setLoggedInEmail(authEmail);
                localStorage.setItem('isLoggedIn', 'true');
                localStorage.setItem('loggedInEmail', authEmail);
                setAuthEmail('');
                setAuthPassword('');
                setAuthSuccess('');
            }
        } catch (error) {
            console.error("Login fetch error:", error);
            setAuthError('Login failed: Network error or unexpected response.');
        }
    };

    const handleSignup = async () => {
        setAuthError('');
        setAuthSuccess('');
        if (!authEmail || !authPassword) {
            setAuthError('Please enter both email and password.');
            return;
        }
        try {
            const res = await fetch('http://localhost:8000/signup', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email: authEmail, password: authPassword }),
            });
            const data = await res.json();
            if (!res.ok) {
                setAuthError(data.detail || 'Signup failed');
            } else {
                setAuthSuccess('Signup successful!');
                setTimeout(() => {
                    setIsLoggedIn(true);
                    setLoggedInEmail(authEmail);
                    localStorage.setItem('isLoggedIn', 'true');
                    localStorage.setItem('loggedInEmail', data.email); // Use data.email for consistency
                    setAuthEmail('');
                    setAuthPassword('');
                    setAuthSuccess('');
                }, 1200);
            }
        } catch (error) {
            console.error("Signup fetch error:", error);
            setAuthError('Signup failed: Network error or unexpected response.');
        }
    };

    const handleSignOut = () => {
        setProfileOpen(false);
        setIsLoggedIn(false);
        localStorage.removeItem('isLoggedIn');
        localStorage.removeItem('loggedInEmail');
        sessionStorage.removeItem('activeThreadId');
        setLoggedInEmail('');
        setQuestion('');
        setChat([]);
        setThreads([]);
        setActiveThreadId(null);
        setIsSidebarOpen(false);
        setSelectedFile(null); // Clear selected file on sign out
        setFilePreviewUrl(null); // Clear file preview on sign out
        setFileType(null); // Clear file type on sign out
        setIsBotTyping(false);
        setIsWebSearching(false);
        setIsLoadingChat(false); // NEW: Ensure chat loading is off on sign out
        setTimeout(() => setSignOutMsg('Successfully signed out'), 100);
        setTimeout(() => setSignOutMsg(''), 2000);
    };

    // Function to check if query needs web search
    const needsWebSearch = (query) => {
        const webSearchIndicators = [
            'latest', 'recent', 'current', 'today', 'now', 'this week', 'this month', 'this year',
            'updated', 'new', 'breaking', 'fresh', 'this season', 'this time', '2024', '2025',
            'news', 'headlines', 'breaking news', 'current events', 'happening', 'price', 'stock',
            'weather', 'temperature', 'forecast', 'exchange rate', 'currency', 'bitcoin', 'crypto',
            'search for', 'find information about', 'look up', 'google', 'what is happening',
            'what\'s new', 'tell me about recent', 'trending', 'viral', 'popular',
            'top rated', 'best of', 'review', 'comparison', 'versus', 'vs',
            'winner', 'champion', 'championship', 'tournament', 'final', 'match result',
            'ipl', 'world cup', 'olympics', 'premier league', 'nba', 'nfl', 'fifa',
            'score', 'result', 'standings', 'league table', 'season',
            'who is', 'what is', 'where is', 'when did', 'how to', 'who won', 'who wins',
            'status of', 'information about', 'details about'
        ];

        const excludeIndicators = [
            'uploaded file', 'this file', 'document', 'pdf', 'above file',
            'according to', 'based on the file', 'from the document',
            'dvdrental', 'database', 'table', 'sql', 'query database'
        ];

        const queryLower = query.toLowerCase();

        // Check exclusions first
        for (const exclude of excludeIndicators) {
            if (queryLower.includes(exclude)) return false;
        }

        // Check web search indicators
        for (const indicator of webSearchIndicators) {
            if (queryLower.includes(indicator)) return true;
        }

        // Additional pattern matching for sports and time-sensitive queries
        const patterns = [
            /winner.*season/i,
            /champion.*\d{4}/i,
            /ipl.*season/i,
            /who won.*tournament/i,
            /result.*match/i,
            /^who is.*(?!in|from|mentioned|discussed)/i,
            /^who.*(winner|champion|won|wins)/i
        ];

        for (const pattern of patterns) {
            if (pattern.test(query)) return true;
        }

        return false;
    };

    const handleSend = async () => {
        if (!question.trim() && !selectedFile) return;

        const userText = question;

        // Check if this query needs web search
        const requiresWebSearch = needsWebSearch(userText);

        // Add user message to chat state immediately
        setChat(prev => [...prev, {
            type: 'user',
            text: userText,
            image_data_base64: fileType === 'image' && filePreviewUrl ? filePreviewUrl.split(',')[1] : null,
            image_mime_type: fileType === 'image' ? selectedFile?.type : null,
            filename: selectedFile?.name,
            video_data_base664: fileType === 'video' && filePreviewUrl ? filePreviewUrl.split(',')[1] : null,
            video_mime_type: fileType === 'video' ? selectedFile?.type : null, // Corrected to use selectedFile.type
        }]);

        // Clear input and file state immediately after adding to chat
        setQuestion('');
        const currentFile = selectedFile; // Store reference before clearing
        setSelectedFile(null);
        setFilePreviewUrl(null);
        setFileType(null);
        // Clear the file input value to allow re-selecting the same file
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }

        // Set appropriate loading indicator
        if (requiresWebSearch) {
            setIsWebSearching(true);
        } else {
            setIsBotTyping(true);
        }

        const formData = new FormData();
        formData.append('question', userText);
        formData.append('model', selectedModel);
        if (activeThreadId) {
            formData.append('thread_id', activeThreadId.toString());
        }
        if (currentFile) {
            formData.append('file', currentFile);
        }

        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: {
                    'X-User-Email': loggedInEmail
                },
                body: formData,
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.detail || `HTTP error! status: ${response.status}`);
            }

            setChat(prev => [...prev, {
                type: 'bot',
                text: data.answer,
                image_data_base64: data.image_data_base64 || null,
                image_mime_type: data.image_mime_type || null,
                video_data_base64: data.video_data_base64 || null,
                video_mime_type: data.video_mime_type || null,
                filename: data.filename || null,
                websearch_info: data.websearch_info || null,
            }]);

            if (!activeThreadId) {
                setActiveThreadId(data.thread_id);
                sessionStorage.setItem('activeThreadId', data.thread_id.toString());
                fetch('http://localhost:8000/threads', {
                    headers: { "X-User-Email": loggedInEmail }
                })
                    .then(res => res.json())
                    .then(updatedThreads => {
                        setThreads(updatedThreads.map(thread => ({ ...thread, isLoading: false })));
                    })
                    .catch(error => console.error("Failed to re-fetch threads after new chat:", error));
            }
        } catch (error) {
            console.error("Error communicating with chatbot:", error);
            setChat(prev => [...prev, { type: 'bot', text: `❌ Error: ${error.message}`, image_data_base64: null, video_data_base64: null }]);
        } finally {
            setIsBotTyping(false);
            setIsWebSearching(false);
            // Ensure file state is completely cleared after request completes
            setSelectedFile(null);
            setFilePreviewUrl(null);
            setFileType(null);
            if (fileInputRef.current) {
                fileInputRef.current.value = "";
            }
            inputRef.current?.focus();
        }
    };

    const handleNewChat = () => {
        // Reset all relevant states for a new chat
        setActiveThreadId(null);
        sessionStorage.removeItem('activeThreadId');
        setChat([]);
        setQuestion('');
        setSelectedFile(null);
        setFilePreviewUrl(null);
        setFileType(null);
        setIsBotTyping(false); // Ensure typing indicator is off for new chat
        setIsWebSearching(false); // Ensure web search indicator is off for new chat
        setIsLoadingChat(false); // NEW: Ensure chat loading is off for a new chat
        // Ensure no thread is marked as loading when starting a new chat
        setThreads(prevThreads => prevThreads.map(thread => ({ ...thread, isLoading: false })));
        inputRef.current?.focus();
    };

    const handleProfileClick = () => setProfileOpen(v => !v);

    const handleShowSignup = () => {
        setShowSignup(true);
        setAuthError('');
        setAuthSuccess('');
        setAuthEmail('');
        setAuthPassword('');
    };

    const handleBackToLogin = () => {
        setShowSignup(false);
        setAuthError('');
        setAuthSuccess('');
        setAuthEmail('');
        setAuthPassword('');
    };

    const handleGoogleSuccess = async (credentialResponse) => {
        const idToken = credentialResponse.credential;
        try {
            const res = await fetch('http://localhost:8000/auth/google', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id_token: idToken }),
            });
            const data = await res.json();
            if (!res.ok) {
                setAuthError(data.detail || 'Google authentication failed');
            } else {
                setIsLoggedIn(true);
                setLoggedInEmail(data.email);
                localStorage.setItem('isLoggedIn', 'true');
                localStorage.setItem('loggedInEmail', data.email);
                setAuthEmail('');
                setAuthPassword('');
                setAuthSuccess('');
                setAuthError('');
            }
        } catch (error) {
            console.error("Google auth fetch error:", error);
            setAuthError('Google authentication failed: Network error or unexpected response.');
        }
    };

    const handleGoogleError = () => {
        setAuthError('Google authentication failed. Please try again.');
    };

    const getProfileInitial = () => {
        if (loggedInEmail && typeof loggedInEmail === 'string' && loggedInEmail.length > 0) {
            return loggedInEmail[0].toUpperCase();
        }
        return '';
    };

    const handleDeleteThread = async (threadIdToDelete) => {
        // Replaced window.confirm with a custom modal in a real app.
        // For this example, keeping window.confirm as per original code structure.
        if (window.confirm("Are you sure you want to delete this chat?")) {
            try {
                await fetch(`http://localhost:8000/threads/${threadIdToDelete}`, {
                    method: 'DELETE',
                    headers: { 'X-User-Email': loggedInEmail },
                });
                setThreads(prev => prev.filter(t => t.id !== threadIdToDelete));
                if (activeThreadId === threadIdToDelete) {
                    setActiveThreadId(null);
                    sessionStorage.removeItem('activeThreadId');
                    setChat([]);
                    setIsLoadingChat(false); // NEW: Ensure no chat loading after deletion
                }
            } catch (error) {
                console.error("Failed to delete thread:", error);
            }
        }
        setOptionsMenuThreadId(null);
    };

    const handleStartEditing = (thread) => {
        setOptionsMenuThreadId(null);
        setEditingThreadId(thread.id);
        setEditingTitle(thread.title);
    };

    const handleFinishEditing = async (threadId) => {
        if (!editingTitle.trim()) {
            setEditingThreadId(null);
            return;
        }
        try {
            await fetch(`http://localhost:8000/threads/${threadId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'X-User-Email': loggedInEmail,
                },
                body: JSON.stringify({ title: editingTitle }),
            });
            setThreads(prev => prev.map(t => t.id === threadId ? { ...t, title: editingTitle } : t));
        } catch (error) {
            console.error("Failed to update thread title:", error);
        } finally {
            setEditingThreadId(null);
        }
    };

    const handlePinThread = async (threadToPin) => {
        try {
            await fetch(`http://localhost:8000/threads/${threadToPin.id}/pin`, {
                method: 'PUT',
                headers: { 'X-User-Email': loggedInEmail },
            });
            fetch('http://localhost:8000/threads', {
                headers: { "X-User-Email": loggedInEmail }
            })
                .then(res => res.json())
                .then(data => setThreads(data.map(thread => ({ ...thread, isLoading: false })))) // Ensure pinned threads also have isLoading
                .catch(error => console.error("Failed to re-fetch threads after pin:", error));
        } catch (error) {
            console.error("Failed to pin thread:", error);
        } finally {
            setOptionsMenuThreadId(null);
        }
    };

    const handleThemeChange = (newTheme) => {
        setTheme(newTheme);
        // setShowThemeSelector(false); // Assuming this state exists if you have a theme selector dropdown
    };

    // Handler for file input change (updated for PDF with name and format)
    const handleFileUpload = (event) => {
        const file = event.target.files[0];
        if (file) {
            // Clear any existing file state first
            setSelectedFile(null);
            setFilePreviewUrl(null);
            setFileType(null);

            // Small delay to ensure state is cleared before setting new file
            setTimeout(() => {
                setSelectedFile(file);
                // Determine file type for preview and mime type
                if (file.type.startsWith('image/')) {
                    setFileType('image');
                    const reader = new FileReader();
                    reader.onloadend = () => {
                        setFilePreviewUrl(reader.result); // Base64 for images (data URL)
                    };
                    reader.readAsDataURL(file);
                } else if (file.type === 'application/pdf') {
                    setFileType('pdf');
                    setFilePreviewUrl('pdf-selected');
                } else if (file.type.startsWith('video/')) {
                    setFileType('video');
                    const reader = new FileReader();
                    reader.onloadend = () => {
                        setFilePreviewUrl(reader.result); // Base64 for videos (data URL)
                    };
                    reader.readAsDataURL(file);
                } else if (file.type.includes('wordprocessingml') || file.type === 'application/msword' || file.name.toLowerCase().endsWith('.doc') || file.name.toLowerCase().endsWith('.docx')) {
                    setFileType('document');
                    setFilePreviewUrl('document-selected');
                } else if (file.type.includes('presentationml') || file.type === 'application/vnd.ms-powerpoint' || file.name.toLowerCase().endsWith('.ppt') || file.name.toLowerCase().endsWith('.pptx')) {
                    setFileType('presentation');
                    setFilePreviewUrl('presentation-selected');
                } else if (file.type.includes('spreadsheetml') || file.type === 'application/vnd.ms-excel' || file.name.toLowerCase().endsWith('.xls') || file.name.toLowerCase().endsWith('.xlsx')) {
                    setFileType('spreadsheet');
                    setFilePreviewUrl('spreadsheet-selected');
                } else if (file.type === 'text/csv' || file.type === 'application/csv' || file.name.toLowerCase().endsWith('.csv')) {
                    setFileType('csv');
                    setFilePreviewUrl('csv-selected');
                } else if (file.type === 'text/plain' || file.name.toLowerCase().endsWith('.txt')) {
                    setFileType('text');
                    setFilePreviewUrl('text-selected');
                } else if (file.type === 'application/json' || file.name.toLowerCase().endsWith('.json')) {
                    setFileType('json');
                    setFilePreviewUrl('json-selected');
                } else if (file.type === 'text/markdown' || file.name.toLowerCase().endsWith('.md')) {
                    setFileType('markdown');
                    setFilePreviewUrl('markdown-selected');
                } else {
                    setFileType('other');
                    setFilePreviewUrl(null); // No direct preview for other file types
                }
            }, 10);
        }
    };

    // Helper function to get file type and display info
    const getFileDisplayInfo = (filename) => {
        if (!filename) return null;

        const ext = filename.toLowerCase().split('.').pop();

        switch (ext) {
            case 'pdf':
                return {
                    type: 'PDF Document',
                    color: '#EF4444',
                    bgColor: '#FEF2F2',
                    borderColor: '#FECACA',
                    badgeColor: '#EF4444',
                    label: 'PDF'
                };
            case 'docx':
            case 'doc':
                return {
                    type: 'Word Document',
                    color: '#2563EB',
                    bgColor: '#EFF6FF',
                    borderColor: '#DBEAFE',
                    badgeColor: '#2563EB',
                    label: 'DOC'
                };
            case 'pptx':
            case 'ppt':
                return {
                    type: 'PowerPoint Presentation',
                    color: '#EA580C',
                    bgColor: '#FFF7ED',
                    borderColor: '#FED7AA',
                    badgeColor: '#EA580C',
                    label: 'PPT'
                };
            case 'xlsx':
            case 'xls':
                return {
                    type: 'Excel Spreadsheet',
                    color: '#16A34A',
                    bgColor: '#F0FDF4',
                    borderColor: '#BBF7D0',
                    badgeColor: '#16A34A',
                    label: 'XLS'
                };
            case 'csv':
                return {
                    type: 'CSV Data',
                    color: '#CA8A04',
                    bgColor: '#FEFCE8',
                    borderColor: '#FDE68A',
                    badgeColor: '#CA8A04',
                    label: 'CSV'
                };
            case 'txt':
                return {
                    type: 'Text File',
                    color: '#6B7280',
                    bgColor: '#F9FAFB',
                    borderColor: '#E5E7EB',
                    badgeColor: '#6B7280',
                    label: 'TXT'
                };
            case 'json':
                return {
                    type: 'JSON Data',
                    color: '#6B7280',
                    bgColor: '#F9FAFB',
                    borderColor: '#E5E7EB',
                    badgeColor: '#6B7280',
                    label: 'JSON'
                };
            case 'md':
                return {
                    type: 'Markdown File',
                    color: '#6B7280',
                    bgColor: '#F9FAFB',
                    borderColor: '#E5E7EB',
                    badgeColor: '#6B7280',
                    label: 'MD'
                };
            default:
                return {
                    type: 'Document',
                    color: '#6B7280',
                    bgColor: '#F9FAFB',
                    borderColor: '#E5E7EB',
                    badgeColor: '#6B7280',
                    label: 'FILE'
                };
        }
    };

    // Handler to remove selected file
    const handleRemoveFile = () => {
        setSelectedFile(null);
        setFilePreviewUrl(null);
        setFileType(null);
        // Clear the file input value to allow re-selecting the same file
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    };


    // New bot icon component based on the provided image
    const BotIcon = ({ size = 36, className = "" }) => (
        <svg width={size} height={size} viewBox="0 0 100 100" fill="none" className={className}>
            {/* Main body - light blue/cyan gradient */}
            <ellipse cx="50" cy="60" rx="35" ry="30" fill="url(#bodyGradient)" />

            {/* Head - circular with gradient */}
            <circle cx="50" cy="35" r="25" fill="url(#headGradient)" />

            {/* Headset band */}
            <path d="M25 35 Q50 15 75 35" stroke={currentTheme.primary} strokeWidth="3" fill="none" strokeLinecap="round" />

            {/* Left headphone */}
            <circle cx="25" cy="35" r="8" fill={currentTheme.primary} />
            <circle cx="25" cy="35" r="5" fill="#ffffff" />

            {/* Right headphone */}
            <circle cx="75" cy="35" r="8" fill={currentTheme.primary} />
            <circle cx="75" cy="35" r="5" fill="#ffffff" />

            {/* Microphone */}
            <line x1="25" y1="43" x2="30" y2="50" stroke={currentTheme.primary} strokeWidth="2" fill="none" strokeLinecap="round" />
            <circle cx="30" cy="52" r="2" fill={currentTheme.primary} />

            {/* Face area - darker blue circle */}
            <circle cx="50" cy="35" r="18" fill="#1a237e" />

            {/* Eyes - white rectangles */}
            <rect x="43" y="30" width="4" height="8" rx="2" fill="#ffffff" />
            <rect x="53" y="30" width="4" height="8" rx="2" fill="#ffffff" />

            {/* Smile */}
            <path d="M42 42 Q50 48 58 42" stroke="#ffffff" strokeWidth="2" fill="none" strokeLinecap="round" />

            {/* AI Badge on chest */}
            <rect x="40" y="55" width="20" height="12" rx="6" fill="#1a237e" />
            <text x="50" y="63" textAnchor="middle" fill="#ffffff" fontSize="8" fontWeight="bold">AI</text>

            {/* Arms */}
            <ellipse cx="20" cy="50" rx="8" ry="15" fill={currentTheme.primary} />
            <ellipse cx="80" cy="50" rx="8" ry="15" fill={currentTheme.primary} />

            {/* Gradient definitions */}
            <defs>
                <linearGradient id="headGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#e0f7ff" />
                    <stop offset="100%" stopColor="#b3e5fc" />
                </linearGradient>
                <linearGradient id="bodyGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#e0f7ff" />
                    <stop offset="100%" stopColor="#b3e5fc" />
                </linearGradient>
            </defs>
        </svg>
    );

    // Toggle between dark and light mode
    const handleModeToggle = () => {
        setIsDarkMode(!isDarkMode);
    };

    // Define colors based on mode
    const modeColors = {
        background: isDarkMode ? '#000000' : '#ffffff',
        surface: isDarkMode ? '#171717' : '#f5f5f5',
        surfaceSecondary: isDarkMode ? '#262626' : '#e5e5e5',
        text: isDarkMode ? '#ffffff' : '#000000',
        textSecondary: isDarkMode ? '#a3a3a3' : '#525252',
        border: isDarkMode ? '#404040' : '#d4d4d8'
    };

    // Calculate filtered threads based on search term
    const filteredThreads = threads
        .filter(thread =>
            thread.title && thread.title.toLowerCase().includes(searchTerm.toLowerCase())
        )
        .sort((a, b) => {
            // Sort pinned threads first
            if (a.pinned && !b.pinned) return -1;
            if (!a.pinned && b.pinned) return 1;
            // Then sort by most recent (assuming higher ID is more recent or add a timestamp)
            return b.id - a.id;
        });

    // Update the modelOptions to reflect current API capabilities (with emojis restored)
    const modelOptions = [
        {
            value: "chatgpt",
            label: "🤖 GPT-4o Mini",
            capabilities: "Text • Image • Video" // For display below the selector
        },
        {
            value: "gemini-1.5-flash", // Kept original value
            label: "⚡ Gemini 1.5 Flash",
            capabilities: "Text • Image • Video" // Assuming it is multimodal and can handle video based on context
        },
        {
            value: "gemini-2.5-pro", // Kept original value
            label: "✨ Gemini 2.5 Pro",
            capabilities: "Text • Image" // Assuming it is multimodal
        },
        {
            value: "gemini-2.0-flash-exp", // New Gemini model
            label: "🔬 Gemini 2.0 Flash Exp",
            capabilities: "Text • Image • Video" // Latest Gemini model with full capabilities
        },
        {
            value: "deepseek-v3", // OpenRouter DeepSeek model
            label: "🧠 DeepSeek V3",
            capabilities: "Text • Image • Video" // Full capabilities through OpenRouter
        },
        {
            value: "mistral-nemo", // OpenRouter Mistral model
            label: "🌊 Mistral Nemo",
            capabilities: "Text • Image • Video" // Full capabilities through OpenRouter
        }
    ];

    // Auth UI
    if (!isLoggedIn) {
        if (!GOOGLE_CLIENT_ID) {
            return (
                <div className="min-h-screen flex items-center justify-center bg-red-100 p-4">
                    <div className="bg-white p-8 rounded-lg shadow-lg max-w-md w-full text-center">
                        <h1 className="text-3xl font-bold text-red-800 mb-6">Configuration Error</h1>
                        <p className="text-red-600 mb-8">
                            Google Client ID is missing. Please ensure `VITE_GOOGLE_CLIENT_ID` is set in your `.env` file and restart the development server.
                        </p>
                    </div>
                </div>
            );
        }

        return (
            <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
                <div
                    className="min-h-screen w-screen flex items-center justify-center text-white overflow-hidden"
                    style={{
                        background: `linear-gradient(135deg, ${currentTheme.background} 0%, ${currentTheme.surface} 50%, ${currentTheme.background} 100%)`,
                        ...themeStyles
                    }}
                >
                    <div
                        className="w-full max-w-md mx-4 rounded-2xl shadow-2xl p-6 backdrop-blur-sm"
                        style={{
                            backgroundColor: `${currentTheme.surface}dd`,
                            border: `1px solid ${currentTheme.primary}33`,
                            maxHeight: 'calc(100vh - 4rem)'
                        }}
                    >
                        <div className="flex items-center justify-center gap-4 mb-6">
                            <BotIcon size={48} />
                            <h1
                                className="text-3xl font-bold tracking-wide"
                                style={{ color: currentTheme.primary }}
                            >
                                Kratos
                            </h1>
                        </div>

                        <div className="w-full space-y-4">
                            <h2 className="text-lg font-semibold text-center text-white mb-4">
                                {showSignup ? 'Create Account' : 'Welcome Back'}
                            </h2>

                            {authError && (
                                <div className="text-red-400 text-sm text-center p-2 rounded-lg bg-red-900/20 border border-red-500/30">
                                    {authError}
                                </div>
                            )}

                            {authSuccess && (
                                <div className="text-green-400 text-sm fixed top-8 left-1/2 -translate-x-1/2 bg-green-800/90 backdrop-blur-sm px-6 py-3 rounded-lg shadow-lg z-50 border border-green-500/30">
                                    {authSuccess}
                                </div>
                            )}

                            {!showSignup ? (
                                <>
                                    {/* Login Fields */}
                                    <div className="space-y-1">
                                        <label className="text-sm font-medium text-white/80" htmlFor="login-username">
                                            Email Address
                                        </label>
                                        <div className="relative">
                                            <span
                                                className="absolute left-4 top-1/2 -translate-y-1/2"
                                                style={{ color: currentTheme.primary }}
                                            >
                                                <svg width="18" height="18" fill="none">
                                                    <circle cx="9" cy="6" r="3" fill="currentColor" />
                                                    <ellipse cx="9" cy="14" rx="6" ry="3" fill="currentColor" />
                                                </svg>
                                            </span>
                                            <input
                                                id="login-username"
                                                type="email"
                                                className="w-full rounded-xl pl-11 pr-4 py-3 text-white transition-all duration-200 focus:outline-none focus:ring-2"
                                                style={{
                                                    backgroundColor: currentTheme.surface,
                                                    border: `2px solid ${currentTheme.primary}33`,
                                                    focusRingColor: currentTheme.primary
                                                }}
                                                placeholder="Enter your email"
                                                value={authEmail}
                                                onChange={e => setAuthEmail(e.target.value)}
                                                autoComplete="username"
                                            />
                                        </div>
                                    </div>

                                    <div className="space-y-1">
                                        <label className="text-sm font-medium text-white/80" htmlFor="login-password">
                                            Password
                                        </label>
                                        <div className="relative">
                                            <span
                                                className="absolute left-4 top-1/2 -translate-y-1/2"
                                                style={{ color: currentTheme.primary }}
                                            >
                                                <svg width="18" height="18" fill="none">
                                                    <rect x="3" y="8" width="12" height="7" rx="2" fill="currentColor" />
                                                    <rect x="6" y="4" width="6" height="4" rx="3" fill="currentColor" />
                                                </svg>
                                            </span>
                                            <input
                                                id="login-password"
                                                type={showPassword ? 'text' : 'password'}
                                                className="w-full rounded-xl pl-11 pr-11 py-3 text-white transition-all duration-200 focus:outline-none focus:ring-2"
                                                style={{
                                                    backgroundColor: currentTheme.surface,
                                                    border: `2px solid ${currentTheme.primary}33`,
                                                    focusRingColor: currentTheme.primary
                                                }}
                                                placeholder="Enter your password"
                                                value={authPassword}
                                                onChange={e => setAuthPassword(e.target.value)}
                                                autoComplete="current-password"
                                            />
                                            <button
                                                type="button"
                                                className="absolute right-4 top-1/2 -translate-y-1/2"
                                                onClick={() => setShowPassword(!showPassword)}
                                                style={{ color: currentTheme.primary }}
                                            >
                                                {showPassword ? (
                                                    <svg width="18" height="18" fill="currentColor" viewBox="0 0 20 20">
                                                        <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
                                                        <path fillRule="evenodd" d="M.458 10C3.732 4.943 9.522 3 10 3s6.268 1.943 9.542 7c-3.274 5.057-9.03 7-9.542 7S3.732 15.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd" />
                                                    </svg>
                                                ) : (
                                                    <svg width="18" height="18" fill="currentColor" viewBox="0 0 20 20">
                                                        <path fillRule="evenodd" d="M3.707 2.293a1 1 0 00-1.414 1.414l14 14a1 1 0 001.414-1.414l-1.473-1.473A10.014 10.014 0 0019.542 10C16.268 4.943 10.478 3 10 3a9.958 9.958 0 00-4.512 1.074l-1.78-1.781zm4.261 4.26l1.514 1.515a2.003 2.003 0 012.45 2.45l1.514 1.514a4 4 0 00-5.478-5.478z" clipRule="evenodd" />
                                                        <path d="M12.454 16.697L9.75 13.992a4 4 0 01-3.742-3.741L2.335 6.578A9.98 9.98 0 00.458 10c3.274 5.057 9.03 7 9.542 7 .847 0 1.68-.127 2.454-.369z" />
                                                    </svg>
                                                )}
                                            </button>
                                        </div>
                                    </div>

                                    <button
                                        className="w-full text-white font-semibold py-3 rounded-xl transition-all duration-200 transform hover:scale-105 active:scale-95 mt-4"
                                        style={{
                                            backgroundColor: currentTheme.primary,
                                            boxShadow: `0 8px 25px ${currentTheme.primary}40`
                                        }}
                                        onMouseEnter={e => e.target.style.backgroundColor = currentTheme.primaryHover}
                                        onMouseLeave={e => e.target.style.backgroundColor = currentTheme.primary}
                                        onClick={handleLogin}
                                    >
                                        Sign In
                                    </button>

                                    <div className="relative my-4">
                                        <div className="absolute inset-0 flex items-center">
                                            <div className="w-full border-t border-white/20"></div>
                                        </div>
                                        <div className="relative flex justify-center text-sm">
                                            <span className="px-4 text-white/60" style={{ backgroundColor: currentTheme.surface }}>
                                                Or continue with
                                            </span>
                                        </div>
                                    </div>

                                    <div className="flex justify-center">
                                        <GoogleLogin
                                            onSuccess={handleGoogleSuccess}
                                            onError={handleGoogleError}
                                            width="100%"
                                        />
                                    </div>

                                    <button
                                        className="w-full font-semibold py-2 transition-colors mt-3"
                                        style={{ color: currentTheme.primary }}
                                        onClick={handleShowSignup}
                                    >
                                        Create New Account
                                    </button>
                                </>
                            ) : (
                                <> {/* This Fragment wraps the signup form elements */}
                                    {/* Signup Fields */}
                                    <div className="space-y-1">
                                        <label className="text-sm font-medium text-white/80" htmlFor="signup-username">
                                            Email Address
                                        </label>
                                        <div className="relative">
                                            <span
                                                className="absolute left-4 top-1/2 -translate-y-1/2"
                                                style={{ color: currentTheme.primary }}
                                            >
                                                <svg width="18" height="18" fill="none">
                                                    <circle cx="9" cy="6" r="3" fill="currentColor" />
                                                    <ellipse cx="9" cy="14" rx="6" ry="3" fill="currentColor" />
                                                </svg>
                                            </span>
                                            <input
                                                id="signup-username"
                                                type="email"
                                                className="w-full rounded-xl pl-11 pr-4 py-3 text-white transition-all duration-200 focus:outline-none focus:ring-2"
                                                style={{
                                                    backgroundColor: currentTheme.surface,
                                                    border: `2px solid ${currentTheme.primary}33`
                                                }}
                                                placeholder="Enter your email"
                                                value={authEmail}
                                                onChange={e => setAuthEmail(e.target.value)}
                                                autoComplete="username"
                                            />
                                        </div>
                                    </div>

                                    <div className="space-y-1">
                                        <label className="text-sm font-medium text-white/80" htmlFor="signup-password">
                                            Password
                                        </label>
                                        <div className="relative">
                                            <span
                                                className="absolute left-4 top-1/2 -translate-y-1/2"
                                                style={{ color: currentTheme.primary }}
                                            >
                                                <svg width="18" height="18" fill="none">
                                                    <rect x="3" y="8" width="12" height="7" rx="2" fill="currentColor" />
                                                    <rect x="6" y="4" width="6" height="4" rx="3" fill="currentColor" />
                                                </svg>
                                            </span>
                                            <input
                                                id="signup-password"
                                                type="password"
                                                className="w-full rounded-xl pl-11 pr-4 py-3 text-white transition-all duration-200 focus:outline-none focus:ring-2"
                                                style={{
                                                    backgroundColor: currentTheme.surface,
                                                    border: `2px solid ${currentTheme.primary}33`
                                                }}
                                                placeholder="Create a password"
                                                value={authPassword}
                                                onChange={e => setAuthPassword(e.target.value)}
                                                autoComplete="new-password"
                                            />
                                        </div>
                                    </div>

                                    <div className="space-y-1">
                                        <label className="text-sm font-medium text-white/80" htmlFor="signup-confirm">
                                            Confirm Password
                                        </label>
                                        <div className="relative">
                                            <span
                                                className="absolute left-4 top-1/2 -translate-y-1/2"
                                                style={{ color: currentTheme.primary }}
                                            >
                                                <svg width="18" height="18" fill="none">
                                                    <rect x="3" y="8" width="12" height="7" rx="2" fill="currentColor" />
                                                    <rect x="6" y="4" width="6" height="4" rx="3" fill="currentColor" />
                                                </svg>
                                            </span>
                                            <input
                                                id="signup-confirm"
                                                type="password"
                                                className="w-full rounded-xl pl-11 pr-4 py-3 text-white transition-all duration-200 focus:outline-none focus:ring-2"
                                                style={{
                                                    backgroundColor: currentTheme.surface,
                                                    border: `2px solid ${currentTheme.primary}33`
                                                }}
                                                placeholder="Confirm your password"
                                                value={authPassword}
                                                onChange={e => setAuthPassword(e.target.value)}
                                                autoComplete="new-password"
                                            />
                                        </div>
                                    </div>

                                    <button
                                        className="w-full text-white font-semibold py-3 rounded-xl transition-all duration-200 transform hover:scale-105 active:scale-95 mt-4"
                                        style={{
                                            backgroundColor: currentTheme.primary,
                                            boxShadow: `0 8px 25px ${currentTheme.primary}40`
                                        }}
                                        onMouseEnter={e => e.target.style.backgroundColor = currentTheme.primaryHover}
                                        onMouseLeave={e => e.target.style.backgroundColor = currentTheme.primary}
                                        onClick={handleSignup}
                                    >
                                        Create Account
                                    </button>

                                    <div className="relative my-4">
                                        <div className="absolute inset-0 flex items-center">
                                            <div className="w-full border-t border-white/20"></div>
                                        </div>
                                        <div className="relative flex justify-center text-sm">
                                            <span className="px-4 text-white/60" style={{ backgroundColor: currentTheme.surface }}>
                                                Or continue with
                                            </span>
                                        </div>
                                    </div>

                                    <div className="flex justify-center">
                                        <GoogleLogin
                                            onSuccess={handleGoogleSuccess}
                                            onError={handleGoogleError}
                                            width="100%"
                                        />
                                    </div>

                                    <button
                                        className="w-full font-semibold py-2 transition-colors mt-3"
                                        style={{ color: currentTheme.primary }}
                                        onClick={handleBackToLogin}
                                    >
                                        Back to Sign In
                                    </button>
                                </>
                            )}

                            {/* This div should always be present at the bottom of the auth form */}
                            <div className="text-xs text-white/50 text-center pt-3 border-t border-white/10 mt-4">
                                Access restricted to Amzur employees
                            </div>
                        </div>
                    </div>

                    {signOutMsg && (
                        <div className="fixed bottom-8 left-1/2 -translate-x-1/2 bg-green-700/90 backdrop-blur-sm text-white px-6 py-3 rounded-lg shadow-lg z-50 border border-green-500/30">
                            {signOutMsg}
                        </div>
                    )}
                </div>
            </GoogleOAuthProvider>
        );
    }

    // Determine if the currently active thread is loading (from initial fetch)
    const activeThread = threads.find(thread => thread.id === activeThreadId);
    const isCurrentThreadLoading = activeThread ? activeThread.isLoading : false;

    const handleScrollToBottom = () => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };
    
    // Function to handle file downloads
    const handleFileDownload = (fileData, filename, mimeType) => {
        if (!fileData) {
            console.error("No file data provided for download");
            return;
        }
        
        try {
            // Convert base64 to blob
            const byteCharacters = atob(fileData);
            const byteArrays = [];
            
            for (let offset = 0; offset < byteCharacters.length; offset += 512) {
                const slice = byteCharacters.slice(offset, offset + 512);
                const byteNumbers = new Array(slice.length);
                for (let i = 0; i < slice.length; i++) {
                    byteNumbers[i] = slice.charCodeAt(i);
                }
                const byteArray = new Uint8Array(byteNumbers);
                byteArrays.push(byteArray);
            }
            
            const blob = new Blob(byteArrays, { type: mimeType });
            const url = URL.createObjectURL(blob);
            
            // Create download link
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            
            // Clean up
            setTimeout(() => {
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }, 100);
        } catch (error) {
            console.error("Error downloading file:", error);
        }
    };

    // Chat UI
    return (
        <div
            className="min-h-screen w-screen flex text-white overflow-x-hidden" // Prevent horizontal scroll
            style={{
                backgroundColor: modeColors.background,
                color: modeColors.text,
                ...themeStyles
            }}
        >
            {/* Sidebar */}
            <aside
                ref={sidebarRef}
                // Sidebar width now depends only on isSidebarOpen, with faster transition
                className={`flex flex-col border-r p-4 transition-all duration-100 ${isSidebarOpen ? 'w-64' : 'w-20'} h-screen min-h-0 overflow-x-hidden`}
                style={{
                    backgroundColor: modeColors.surface,
                    borderColor: modeColors.border
                }}
            >
                {/* Menu Bar (top section of sidebar) */}
                <div className={`flex items-center mb-4 ${isSidebarOpen ? 'justify-between' : 'justify-center'}`}>
                    <button
                        onClick={() => setIsSidebarOpen(prev => !prev)}
                        className="p-2 rounded-md transition-colors"
                        style={{
                            backgroundColor: 'transparent', // Default
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = modeColors.surfaceSecondary}
                        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                        id="sidebar-toggle-button" // Added ID for easier targeting in outside click handler
                    >
                        {/* Menu Icon */}
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M3 12h18M3 6h18M3 18h18" />
                        </svg>
                    </button>
                </div>

                {/* Search Bar - only visible when sidebar is open */}
                {isSidebarOpen && (
                    <div className="relative mb-4">
                        <input
                            type="text"
                            placeholder="Search..."
                            className="w-full rounded-md pl-9 pr-3 py-2 text-sm transition-colors focus:outline-none"
                            style={{
                                backgroundColor: modeColors.surfaceSecondary,
                                border: `1px solid ${modeColors.border}`,
                                color: modeColors.text
                            }}
                            value={searchTerm}
                            onChange={e => setSearchTerm(e.target.value)}
                        />
                        <span className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: modeColors.textSecondary }}>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <circle cx="11" cy="11" r="8" />
                                <path d="m21 21-4.35-4.35" />
                            </svg>
                        </span>
                    </div>
                )}

                {/* New Chat Button - adjusted visibility for text */}
                <button
                    onClick={handleNewChat}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg font-semibold transition mb-4 text-white ${isSidebarOpen ? 'justify-start' : 'justify-center'}`}
                    style={{
                        backgroundColor: currentTheme.primary,
                    }}
                    onMouseEnter={e => e.target.style.backgroundColor = currentTheme.primaryHover}
                    onMouseLeave={e => e.target.style.backgroundColor = currentTheme.primary}
                >
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M12 20h9M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
                    </svg>
                    {isSidebarOpen && <span>New Chat</span>} {/* Only show text when open */}
                </button>

                {/* History Section */}
                <div className={`flex-1 min-h-0 ${isSidebarOpen ? 'overflow-y-auto' : 'overflow-hidden'}`}>
                    {/* History Heading - only visible when sidebar is open */}
                    {isSidebarOpen && (
                        <h2 className="text-sm font-semibold mb-2 px-2" style={{ color: modeColors.textSecondary }}>
                            History
                        </h2>
                    )}
                    <nav className="flex flex-col gap-1">
                        {filteredThreads.map(thread => (
                            <div key={thread.id} className="group relative w-full">
                                {editingThreadId === thread.id ? (
                                    <div className="flex items-center gap-2">
                                        <input
                                            type="text"
                                            value={editingTitle}
                                            onChange={(e) => setEditingTitle(e.target.value)}
                                            className="w-full text-left px-3 py-2 rounded-md outline-none border-2"
                                            style={{
                                                backgroundColor: modeColors.surfaceSecondary,
                                                color: modeColors.text,
                                                borderColor: currentTheme.primary
                                            }}
                                            autoFocus
                                            onKeyDown={(e) => {
                                                if (e.key === 'Enter') {
                                                    handleFinishEditing(thread.id);
                                                } else if (e.key === 'Escape') {
                                                    setEditingThreadId(null);
                                                }
                                            }}
                                        />
                                        <button
                                            className="p-1 text-green-400 hover:text-green-600"
                                            title="Save"
                                            onClick={() => handleFinishEditing(thread.id)}
                                        >
                                            <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                                                <polyline points="20 6 9 17 4 12" />
                                            </svg>
                                        </button>
                                        <button
                                            className="p-1 text-red-400 hover:text-red-600"
                                            title="Cancel"
                                            onClick={() => setEditingThreadId(null)}
                                        >
                                            <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                                                <line x1="18" y1="6" x2="6" y2="18" />
                                                <line x1="6" y1="6" x2="18" y2="18" />
                                            </svg>
                                        </button>
                                    </div>
                                ) : (
                                    <div className="flex items-center w-full group relative">
                                        <button
                                            onClick={() => {
                                                // When clicking a thread, set its loading state to true
                                                setThreads(prevThreads =>
                                                    prevThreads.map(t =>
                                                        t.id === thread.id ? { ...t, isLoading: true } : { ...t, isLoading: false } // Reset others
                                                    )
                                                );
                                                setChat([]); // Clear current chat immediately for visual feedback
                                                setActiveThreadId(thread.id);
                                                sessionStorage.setItem('activeThreadId', thread.id.toString());
                                            }}
                                            className={`flex-1 flex items-center text-left px-3 py-2 rounded-md truncate transition ${!isSidebarOpen ? 'justify-center' : ''}`}
                                            style={{
                                                backgroundColor: activeThreadId === thread.id ? modeColors.surfaceSecondary : 'transparent',
                                                color: modeColors.text, // Text color when sidebar is open, but text is hidden when closed
                                            }}
                                            onMouseEnter={e => {
                                                if (activeThreadId !== thread.id) {
                                                    e.target.style.backgroundColor = `${modeColors.surfaceSecondary}80`;
                                                }
                                            }}
                                            onMouseLeave={e => {
                                                if (activeThreadId !== thread.id) {
                                                    e.target.style.backgroundColor = 'transparent';
                                                }
                                            }}
                                            title={isSidebarOpen ? thread.title : (thread.pinned ? 'Pinned Chat' : 'Chat')} // Show title on hover for closed sidebar
                                        >
                                            <div className="flex items-center gap-2 truncate">
                                                {/* Always show pin icon if pinned, regardless of sidebar state, but text only when open */}
                                                {thread.pinned && (
                                                    <svg width="14" height="14" viewBox="0 0 24 24" fill={currentTheme.primary} className="flex-shrink-0">
                                                        <path d="M16 3a1 1 0 0 1 1 1v5.268l2.56 2.56a1 1 0 0 1 .293.707V14a1 1 0 0 1-1 1h-4.268l-2.56 2.56a1 1 0 0 1-.707.293H10a1 1 0 0 1-1-1v-4.268l-2.56-2.56a1 1 0 0 1-.293-.707V8a1 1 0 0 1 1-1h5.268L14 4.434V4a1 1 0 0 1 1-1z" />
                                                    </svg>
                                                )}
                                                {isSidebarOpen && <span className="truncate">{thread.title}</span>} {/* Only show thread title when open */}
                                            </div>
                                        </button>

                                        {/* Ellipsis text (options menu toggle) - only show when sidebar is open and not editing */}
                                        {isSidebarOpen && editingThreadId !== thread.id && (
                                            <div
                                                className={`absolute right-0 top-1/2 -translate-y-1/2 cursor-pointer p-2 transition-colors text-2xl font-bold rounded-full hover:bg-neutral-700/30`}
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    setOptionsMenuThreadId(optionsMenuThreadId === thread.id ? null : thread.id);
                                                }}
                                                style={{
                                                    color: modeColors.textSecondary,
                                                    backgroundColor: 'transparent',
                                                }}
                                                onMouseEnter={(e) => {
                                                    e.currentTarget.style.backgroundColor = modeColors.surfaceSecondary;
                                                    e.currentTarget.style.color = modeColors.text;
                                                }}
                                                onMouseLeave={(e) => {
                                                    e.currentTarget.style.backgroundColor = 'transparent';
                                                    e.currentTarget.style.color = modeColors.textSecondary;
                                                }}
                                                title="More options"
                                            >
                                                ...
                                            </div>
                                        )}
                                    </div>
                                )}

                                {/* Options dropdown menu */}
                                {optionsMenuThreadId === thread.id && isSidebarOpen && editingThreadId !== thread.id && (
                                    <div
                                        ref={optionsMenuRef}
                                        className="absolute right-0 mt-2 w-36 rounded-lg shadow-lg z-20 py-1"
                                        style={{
                                            backgroundColor: modeColors.surface,
                                            border: `1px solid ${modeColors.border}`
                                        }}
                                    >
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                handlePinThread(thread);
                                            }}
                                            className="w-full text-left px-3 py-2 text-sm flex items-center gap-2 transition-colors"
                                            style={{ color: modeColors.text }}
                                            onMouseEnter={e => e.target.style.backgroundColor = modeColors.surfaceSecondary}
                                            onMouseLeave={e => e.target.style.backgroundColor = 'transparent'}
                                        >
                                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                <path d="M16 3a1 1 0 0 1 1 1v5.268l2.56 2.56a1 1 0 0 1 .293.707V14a1 1 0 0 1-1 1h-4.268l-2.56 2.56a1 1 0 0 1-.707.293H10a1 1 0 0 1-1-1v-4.268l-2.56-2.56a1 1 0 0 1-.293-.707V8a1 1 0 0 1 1-1h5.268L14 4.434V4a1 1 0 0 1 1-1z" />
                                            </svg>
                                            {thread.pinned ? 'Unpin' : 'Pin'}
                                        </button>
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                handleStartEditing(thread);
                                            }}
                                            className="w-full text-left px-3 py-2 text-sm flex items-center gap-2 transition-colors"
                                            style={{ color: modeColors.text }}
                                            onMouseEnter={e => e.target.style.backgroundColor = modeColors.surfaceSecondary}
                                            onMouseLeave={e => e.target.style.backgroundColor = 'transparent'}
                                        >
                                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                <path d="M12 20h9M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
                                            </svg>
                                            Rename
                                        </button>
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                handleDeleteThread(thread.id);
                                            }}
                                            className="w-full text-left px-3 py-2 text-sm flex items-center gap-2 text-red-400 transition-colors"
                                            style={{ color: modeColors.text }}
                                            onMouseEnter={e => e.target.style.backgroundColor = modeColors.surfaceSecondary}
                                            onMouseLeave={e => e.target.style.backgroundColor = 'transparent'}
                                        >
                                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                <polyline points="3 6 5 6 21 6" />
                                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                                                <line x1="10" y1="11" x2="10" y2="17" />
                                                <line x1="14" y1="11" x2="14" y2="17" />
                                            </svg>
                                            Delete
                                        </button>
                                    </div>
                                )}
                            </div>
                        ))}
                    </nav>
                </div>
            </aside>

            {/* Main Content */}
            <div className="flex-1 flex flex-col h-screen min-h-0 overflow-x-hidden relative">
                {/* Scroll to bottom button - positioned relative to main content */}
                {showScrollToBottom && (
                    <button
                        onClick={handleScrollToBottom}
                        className="fixed bottom-24 right-8 bg-blue-500 text-white p-3 rounded-full shadow-lg transition-all duration-300 hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 z-50"
                        title="Scroll to bottom"
                    >
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="6 9 12 15 18 9"></polyline>
                        </svg>
                    </button>
                )}
                {/* Header */}
                <header
                    className="flex items-center justify-between px-8 h-20 border-b w-full flex-shrink-0"
                    style={{
                        backgroundColor: modeColors.surface,
                        borderColor: modeColors.border
                    }}
                >
                    <div className="flex items-center gap-4">
                        <BotIcon size={40} />
                        <h1 className="text-2xl font-bold tracking-wide" style={{ color: modeColors.text }}>
                            Kratos
                        </h1>
                    </div>

                    <div className="flex items-center gap-4">
                        {/* Chat Icon - For switching to chat view */}
                        <button
                            onClick={() => setActiveView('chat')}
                            className="p-2 rounded-lg transition-all duration-200 hover:scale-110"
                            style={{
                                backgroundColor: activeView === 'chat' ? modeColors.surfaceSecondary : 'transparent',
                                color: activeView === 'chat' ? currentTheme.primary : modeColors.text,
                                border: `1px solid ${activeView === 'chat' ? currentTheme.primary : 'transparent'}`
                            }}
                            title="Chat"
                        >
                            <ChatIcon active={activeView === 'chat'} />
                        </button>
                        
                        {/* Game Icon - For switching to game view */}
                        <button
                            onClick={() => setActiveView('game')}
                            className="p-2 rounded-lg transition-all duration-200 hover:scale-110"
                            style={{
                                backgroundColor: activeView === 'game' ? modeColors.surfaceSecondary : 'transparent',
                                color: activeView === 'game' ? currentTheme.primary : modeColors.text,
                                border: `1px solid ${activeView === 'game' ? currentTheme.primary : 'transparent'}`
                            }}
                            title="Tic-Tac-Toe Game"
                        >
                            <TicTacToeIcon active={activeView === 'game'} />
                        </button>
                        
                        {/* Black/White Mode Toggle */}
                        <button
                            onClick={handleModeToggle}
                            className="p-2 rounded-lg transition-all duration-200 hover:scale-110"
                            style={{
                                backgroundColor: modeColors.surfaceSecondary,
                                color: modeColors.text,
                                border: `1px solid ${modeColors.border}`
                            }}
                            title={isDarkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
                        >
                            {isDarkMode ? (
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <circle cx="12" cy="12" r="5" />
                                    <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
                                </svg>
                            ) : (
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                                </svg>
                            )}
                        </button>

                        {/* Profile */}
                        <div className="relative" ref={profileRef}>
                            <button
                                className="w-11 h-11 flex items-center justify-center rounded-full transition"
                                style={{
                                    ':hover': { backgroundColor: modeColors.surfaceSecondary }
                                }}
                                onMouseEnter={e => e.target.style.backgroundColor = modeColors.surfaceSecondary}
                                onMouseLeave={e => e.target.style.backgroundColor = 'transparent'}
                                onClick={handleProfileClick}
                            >
                                <span
                                    className="w-9 h-9 flex items-center justify-center rounded-full text-white font-bold text-lg uppercase"
                                    style={{ backgroundColor: currentTheme.primary }}
                                >
                                    {getProfileInitial()}
                                </span>
                            </button>
                            {profileOpen && (
                                <div
                                    className="absolute right-0 mt-2 w-56 rounded-lg shadow-lg z-50 py-4 flex flex-col items-center"
                                    style={{
                                        backgroundColor: modeColors.surface,
                                        border: `1px solid ${modeColors.border}`
                                    }}
                                >
                                    {/* User email */}
                                    <div
                                        className="w-full flex justify-center px-4 pb-2 text-sm font-semibold border-b"
                                        style={{
                                            wordBreak: 'break-all',
                                            overflowWrap: 'break-word',
                                            color: modeColors.text,
                                            borderColor: modeColors.border
                                        }}
                                    >
                                        <span
                                            className="inline-block text-center break-all"
                                            style={{
                                                maxWidth: '200px',
                                                whiteSpace: 'normal',
                                                overflowWrap: 'break-word',
                                                wordBreak: 'break-all',
                                            }}
                                            title={loggedInEmail}
                                        >
                                            {loggedInEmail}
                                        </span>
                                    </div>
                                    {/* Profile icon */}
                                    <div className="my-4 flex items-center justify-center">
                                        <span
                                            className="w-14 h-14 flex items-center justify-center rounded-full text-white font-bold text-2xl uppercase"
                                            style={{ backgroundColor: currentTheme.primary }}
                                        >
                                            {getProfileInitial()}
                                        </span>
                                    </div>
                                    {/* Sign Out button with icon */}
                                    <button
                                        className="w-40 flex items-center gap-2 justify-center px-5 py-3 mt-2 rounded-lg transition font-semibold"
                                        style={{
                                            backgroundColor: modeColors.surfaceSecondary,
                                            color: modeColors.text,
                                            ':hover': { backgroundColor: modeColors.border }
                                        }}
                                        onMouseEnter={e => e.target.style.backgroundColor = modeColors.border}
                                        onMouseLeave={e => e.target.style.backgroundColor = modeColors.surfaceSecondary}
                                        onClick={handleSignOut}
                                    >
                                        <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                                            <path d="M17 16l4-4m0 0l-4-4m4 4H7" />
                                            <path d="M7 21a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v2" />
                                        </svg>
                                        Sign Out
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                </header>

                {/* Main Content Area - Conditionally render Chat or Game */}
                {activeView === 'chat' ? (
                    <main
                        ref={chatContainerRef} // Assign ref to the main chat container
                        className="flex-1 flex flex-col w-full min-h-0 overflow-y-auto relative" // Added 'relative' for scroll button positioning
                        style={{ backgroundColor: modeColors.background }}
                    >
                        <div className="flex flex-col gap-4 w-full max-w-5xl mx-auto py-10 px-2 sm:px-8 min-h-0">
                            {/* Conditional rendering for chat messages based on isLoadingChat */}
                            {isLoadingChat ? (
                                <div className="flex w-full justify-start">
                                    <div
                                        className="rounded-xl px-6 py-4 shadow rounded-bl-md max-w-full break-words whitespace-pre-wrap w-full flex items-center gap-2"
                                        style={{
                                            backgroundColor: modeColors.surface,
                                            color: modeColors.text,
                                            border: `1px solid ${modeColors.border}`
                                        }}
                                    >
                                        <span className="animate-pulse text-gray-500">Loading chat...</span>
                                    </div>
                                </div>
                            ) : (
                                <>
                                    {chat.length === 0 && (activeThreadId === null || (activeThread && !activeThread.isLoading)) && (
                                        <div className="w-full flex justify-center items-center h-40">
                                            <span className="text-2xl" style={{ color: modeColors.textSecondary }}>
                                                How can I help you?
                                            </span>
                                        </div>
                                    )}
                                    {chat.map((msg, idx) => (
                                    <div
                                        key={msg.id || idx}
                                        className={`
                                            flex w-full
                                            ${msg.type === 'user' ? 'justify-end' : 'justify-start'}
                                        `}
                                        style={{
                                            backgroundColor: 'transparent',
                                        }}
                                    >
                                        <div
                                            className={`
                                                rounded-xl px-6 py-4 shadow
                                                ${msg.type === 'user'
                                                    ? 'text-white rounded-br-md'
                                                    : 'rounded-bl-md'}
                                                max-w-full break-words w-auto
                                            `}
                                            style={{
                                                maxWidth: '100%',
                                                backgroundColor: msg.type === 'user' ? currentTheme.primary : modeColors.surface,
                                                color: msg.type === 'user' ? '#ffffff' : modeColors.text,
                                                border: msg.type === 'bot' ? `1px solid ${modeColors.border}` : 'none'
                                            }}
                                        >
                                            {/* Document file preview in chat messages - only show for non-image, non-video files */}
                                            {msg.filename && !msg.image_data_base64 && !msg.video_data_base64 && (() => {
                                                const fileInfo = getFileDisplayInfo(msg.filename);
                                                if (!fileInfo) return null;

                                                return (
                                                    <div className="flex items-center gap-2 mb-4 mt-1 p-3 rounded-lg" style={{
                                                        backgroundColor: `${fileInfo.color}1A`, // 10% opacity
                                                        border: `1px solid ${fileInfo.color}33` // 20% opacity
                                                    }}>
                                                        <div className="relative rounded-md p-1 flex items-center justify-center" style={{
                                                            width: '48px',
                                                            height: '48px',
                                                            backgroundColor: fileInfo.bgColor,
                                                            border: `1px solid ${fileInfo.borderColor}`
                                                        }}>
                                                            <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor" style={{ color: fileInfo.color }}>
                                                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14 2z" opacity="0.8" />
                                                                <polyline points="14,2 14,8 20,8" fill="none" stroke="white" strokeWidth="1.5" />
                                                                <text x="12" y="16" textAnchor="middle" fontSize="6" fill="white" fontWeight="bold">PDF</text>
                                                            </svg>
                                                            <div className="absolute -top-1 -right-1 text-white text-xs rounded-full px-1 py-0.5 leading-none font-bold" style={{
                                                                fontSize: '8px',
                                                                backgroundColor: fileInfo.badgeColor
                                                            }}>
                                                                {fileInfo.label}
                                                            </div>
                                                        </div>
                                                        <div className="flex flex-col flex-1">
                                                            <span className="text-sm font-medium truncate max-w-[200px]" style={{ color: modeColors.text }}>
                                                                {msg.filename}
                                                            </span>
                                                            <span className="text-xs opacity-70" style={{ color: modeColors.textSecondary }}>
                                                                {fileInfo.type}
                                                            </span>
                                                        </div>
                                                    </div>
                                                );
                                            })()}

                                            {/* Regular message content with image handling */}
                                            <MessageRenderer
                                                content={msg.text}
                                                imageData={msg.image_mime_type && msg.image_mime_type.startsWith('image/') ? msg.image_data_base64 : null}
                                                videoData={msg.video_data_base64}
                                                imageType={msg.image_mime_type}
                                                videoType={msg.video_mime_type}
                                                websearchInfo={msg.websearch_info}
                                                fileInfo={msg.file_info}
                                                filename={msg.filename}
                                                onDownload={handleFileDownload}
                                                theme={currentTheme}
                                                modeColors={modeColors}
                                            />
                                        </div>
                                    </div>
                                ))}
                                {/* Display typing indicator only if the bot is typing */}
                                {isBotTyping && (
                                    <div className="flex w-full justify-start">
                                        <div
                                            className="rounded-xl px-6 py-4 shadow rounded-bl-md max-w-full break-words whitespace-pre-wrap w-full flex items-center gap-2"
                                            style={{
                                                backgroundColor: modeColors.surface,
                                                color: modeColors.text,
                                                border: `1px solid ${modeColors.border}`
                                            }}
                                        >
                                            <span className="animate-bounce">Typing</span>
                                            <span className="animate-bounce delay-100">.</span>
                                            <span className="animate-bounce delay-200">.</span>
                                            <span className="animate-bounce delay-300">.</span>
                                        </div>
                                    </div>
                                )}
                                {/* Display web search indicator only if web search is active */}
                                {isWebSearching && (
                                    <div className="flex w-full justify-start">
                                        <div
                                            className="rounded-xl px-6 py-4 shadow rounded-bl-md max-w-full break-words whitespace-pre-wrap w-full flex items-center gap-2"
                                            style={{
                                                backgroundColor: modeColors.surface,
                                                color: modeColors.text,
                                                border: `1px solid ${modeColors.border}`
                                            }}
                                        >
                                            <span className="animate-bounce">🔍 Searching the web</span>
                                            <span className="animate-bounce delay-100">.</span>
                                            <span className="animate-bounce delay-200">.</span>
                                            <span className="animate-bounce delay-300">.</span>
                                        </div>
                                    </div>
                                )}
                            </>
                        )}
                        <div ref={chatEndRef} />
                    </div>
                </main>
                ) : (
                    // Tic-Tac-Toe Game View
                    <main 
                        className="flex-1 flex flex-col w-full min-h-0 overflow-y-auto relative"
                        style={{ backgroundColor: modeColors.background }}
                    >
                        <TicTacToeGame />
                    </main>
                )}

                {/* Input Area - Only show when chat view is active */}
                {activeView === 'chat' && (
                    /* Input Area - Redesigned with no border */
                <div
                    className="w-full flex items-center justify-center px-8 py-4"
                    style={{
                        backgroundColor: modeColors.surface,
                        boxShadow: "0 -2px 10px rgba(0,0,0,0.2)"
                    }}
                >
                    <div className="flex flex-col w-full max-w-5xl mx-auto"> {/* Increased max-w to 5xl */}
                        {/* Input container with all controls */}
                        <div className="relative flex items-end rounded-xl p-1 shadow-lg"
                            style={{
                                backgroundColor: modeColors.surfaceSecondary,
                                minHeight: "56px",
                                border: 'none', // Explicitly ensure no border
                            }}
                        >
                            {/* Model selector dropdown */}
                            <div className="relative flex-shrink-0 ml-2 mb-2 self-end" style={{ width: '180px' }}> {/* Set a fixed width to 180px */}
                                <select
                                    className="appearance-none bg-transparent pl-2 pr-8 py-2 text-sm font-medium focus:outline-none cursor-pointer rounded-md w-full"
                                    style={{ color: modeColors.text, backgroundColor: modeColors.surfaceSecondary }}
                                    value={selectedModel}
                                    onChange={(e) => setSelectedModel(e.target.value)}
                                    disabled={isBotTyping || isLoadingChat} // Disable if bot is typing OR chat is loading
                                >
                                    {modelOptions.map(opt => (
                                        <option key={opt.value} value={opt.value} className="bg-neutral-800 py-2">
                                            {opt.label}
                                        </option>
                                    ))}
                                </select>
                                {/* Dropdown arrow */}
                                <span className="absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none" style={{ color: modeColors.textSecondary }}>
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <polyline points="6 9 12 15 18 9"></polyline>
                                    </svg>
                                </span>
                            </div>

                            {/* File preview within the input area */}
                            {selectedFile && (fileType === 'image' || fileType === 'pdf' || fileType === 'video' || fileType === 'document' || fileType === 'presentation' || fileType === 'spreadsheet' || fileType === 'csv' || fileType === 'text' || fileType === 'json' || fileType === 'markdown' || fileType === 'other') && (
                                <div className="flex items-center gap-2 px-3 py-2 rounded-lg border border-blue-400/20 bg-blue-900/10 self-center max-w-[calc(100%-240px)] ml-2">
                                    {fileType === 'image' && filePreviewUrl && (
                                        <img src={filePreviewUrl} alt="Preview" className="max-h-12 w-auto rounded-md object-cover flex-shrink-0" />
                                    )}
                                    {fileType === 'pdf' && (
                                        <div className="flex gap-2 items-center flex-shrink-0">
                                            {/* PDF Thumbnail Icon */}
                                            <div className="relative bg-red-50 border border-red-200 rounded-md p-1 flex items-center justify-center" style={{
                                                width: '48px',
                                                height: '48px',
                                                backgroundColor: '#FEF2F2', // fileInfo.bgColor
                                                border: `1px solid #FECACA` // fileInfo.borderColor
                                            }}>
                                                <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor" className="text-red-600">
                                                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14 2z" fill="#EF4444" />
                                                    <polyline points="14,2 14,8 20,8" fill="none" stroke="white" strokeWidth="1.5" />
                                                    <text x="12" y="16" textAnchor="middle" fontSize="6" fill="white" fontWeight="bold">PDF</text>
                                                </svg>
                                                <div className="absolute -top-1 -right-1 text-white text-xs rounded-full px-1 py-0.5 leading-none font-bold" style={{
                                                    fontSize: '8px',
                                                    backgroundColor: '#EF4444' // fileInfo.badgeColor
                                                }}>
                                                    PDF
                                                </div>
                                            </div>
                                            <div className="flex flex-col">
                                                <span className="text-sm font-medium truncate max-w-[120px]" style={{ color: modeColors.text }}>
                                                    {selectedFile.name}
                                                </span>
                                                <span className="text-xs opacity-70" style={{ color: modeColors.textSecondary }}>
                                                    PDF Document • {(selectedFile.size / 1024).toFixed(1)} KB
                                                </span>
                                            </div>
                                        </div>
                                    )}
                                    {fileType === 'document' && (
                                        <div className="flex gap-2 items-center flex-shrink-0">
                                            {/* Word Document Icon */}
                                            <div className="relative bg-blue-50 border border-blue-200 rounded-md p-1 flex items-center justify-center" style={{
                                                width: '48px',
                                                height: '48px',
                                                backgroundColor: '#EFF6FF', // fileInfo.bgColor
                                                border: `1px solid #DBEAFE` // fileInfo.borderColor
                                            }}>
                                                <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor" className="text-blue-600">
                                                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14 2z" fill="#2563EB" />
                                                    <polyline points="14,2 14,8 20,8" fill="none" stroke="white" strokeWidth="1.5" />
                                                    <text x="12" y="16" textAnchor="middle" fontSize="5" fill="white" fontWeight="bold">DOC</text>
                                                </svg>
                                                <div className="absolute -top-1 -right-1 text-white text-xs rounded-full px-1 py-0.5 leading-none font-bold" style={{ fontSize: '8px', backgroundColor: '#2563EB' }}>
                                                    DOC
                                                </div>
                                            </div>
                                            <div className="flex flex-col">
                                                <span className="text-sm font-medium truncate max-w-[120px]" style={{ color: modeColors.text }}>
                                                    {selectedFile.name}
                                                </span>
                                                <span className="text-xs opacity-70" style={{ color: modeColors.textSecondary }}>
                                                    Word Document • {(selectedFile.size / 1024).toFixed(1)} KB
                                                </span>
                                            </div>
                                        </div>
                                    )}
                                    {fileType === 'presentation' && (
                                        <div className="flex gap-2 items-center flex-shrink-0">
                                            {/* PowerPoint Icon */}
                                            <div className="relative bg-orange-50 border border-orange-200 rounded-md p-1 flex items-center justify-center" style={{
                                                width: '48px',
                                                height: '48px',
                                                backgroundColor: '#FFF7ED', // fileInfo.bgColor
                                                border: `1px solid #FED7AA` // fileInfo.borderColor
                                            }}>
                                                <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor" className="text-orange-600">
                                                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0  0 0 2 2h12a2 2 0 0 0 2-2V7.5L14 2z" fill="#EA580C" />
                                                    <polyline points="14,2 14,8 20,8" fill="none" stroke="white" strokeWidth="1.5" />
                                                    <text x="12" y="16" textAnchor="middle" fontSize="5" fill="white" fontWeight="bold">PPT</text>
                                                </svg>
                                                <div className="absolute -top-1 -right-1 text-white text-xs rounded-full px-1 py-0.5 leading-none font-bold" style={{ fontSize: '8px', backgroundColor: '#EA580C' }}>
                                                    PPT
                                                </div>
                                            </div>
                                            <div className="flex flex-col">
                                                <span className="text-sm font-medium truncate max-w-[120px]" style={{ color: modeColors.text }}>
                                                    {selectedFile.name}
                                                </span>
                                                <span className="text-xs opacity-70" style={{ color: modeColors.textSecondary }}>
                                                    PowerPoint • {(selectedFile.size / 1024).toFixed(1)} KB
                                                </span>
                                            </div>
                                        </div>
                                    )}
                                    {fileType === 'spreadsheet' && (
                                        <div className="flex gap-2 items-center flex-shrink-0">
                                            {/* Excel Icon */}
                                            <div className="relative bg-green-50 border border-green-200 rounded-md p-1 flex items-center justify-center" style={{
                                                width: '48px',
                                                height: '48px',
                                                backgroundColor: '#F0FDF4', // fileInfo.bgColor
                                                border: `1px solid #BBF7D0` // fileInfo.borderColor
                                            }}>
                                                <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor" className="text-green-600">
                                                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14 2z" fill="#16A34A" />
                                                    <polyline points="14,2 14,8 20,8" fill="none" stroke="white" strokeWidth="1.5" />
                                                    <text x="12" y="16" textAnchor="middle" fontSize="5" fill="white" fontWeight="bold">XLS</text>
                                                </svg>
                                                <div className="absolute -top-1 -right-1 text-white text-xs rounded-full px-1 py-0.5 leading-none font-bold" style={{ fontSize: '8px', backgroundColor: '#16A34A' }}>
                                                    XLS
                                                </div>
                                            </div>
                                            <div className="flex flex-col">
                                                <span className="text-sm font-medium truncate max-w-[120px]" style={{ color: modeColors.text }}>
                                                    {selectedFile.name}
                                                </span>
                                                <span className="text-xs opacity-70" style={{ color: modeColors.textSecondary }}>
                                                    Excel Spreadsheet • {(selectedFile.size / 1024).toFixed(1)} KB
                                                </span>
                                            </div>
                                        </div>
                                    )}
                                    {fileType === 'csv' && (
                                        <div className="flex gap-2 items-center flex-shrink-0">
                                            {/* CSV Icon */}
                                            <div className="relative bg-yellow-50 border border-yellow-200 rounded-md p-1 flex items-center justify-center" style={{
                                                width: '48px',
                                                height: '48px',
                                                backgroundColor: '#FEFCE8', // fileInfo.bgColor
                                                border: `1px solid #FDE68A` // fileInfo.borderColor
                                            }}>
                                                <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor" className="text-yellow-600">
                                                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14 2z" fill="#CA8A04" />
                                                    <polyline points="14,2 14,8 20,8" fill="none" stroke="white" strokeWidth="1.5" />
                                                    <text x="12" y="16" textAnchor="middle" fontSize="5" fill="white" fontWeight="bold">CSV</text>
                                                </svg>
                                                <div className="absolute -top-1 -right-1 text-white text-xs rounded-full px-1 py-0.5 leading-none font-bold" style={{ fontSize: '8px', backgroundColor: '#CA8A04' }}>
                                                    CSV
                                                </div>
                                            </div>
                                            <div className="flex flex-col">
                                                <span className="text-sm font-medium truncate max-w-[120px]" style={{ color: modeColors.text }}>
                                                    {selectedFile.name}
                                                </span>
                                                <span className="text-xs opacity-70" style={{ color: modeColors.textSecondary }}>
                                                    CSV Data • {(selectedFile.size / 1024).toFixed(1)} KB
                                                </span>
                                            </div>
                                        </div>
                                    )}
                                    {(fileType === 'text' || fileType === 'json' || fileType === 'markdown') && (
                                        <div className="flex gap-2 items-center flex-shrink-0">
                                            {/* Text/JSON/Markdown Icon */}
                                            <div className="relative bg-gray-50 border border-gray-200 rounded-md p-1 flex items-center justify-center" style={{
                                                width: '48px',
                                                height: '48px',
                                                backgroundColor: '#F9FAFB', // fileInfo.bgColor
                                                border: `1px solid #E5E7EB` // fileInfo.borderColor
                                            }}>
                                                <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor" className="text-gray-600">
                                                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14 2z" fill="#6B7280" />
                                                    <polyline points="14,2 14,8 20,8" fill="none" stroke="white" strokeWidth="1.5" />
                                                    <text x="12" y="16" textAnchor="middle" fontSize="4" fill="white" fontWeight="bold">
                                                        {fileType === 'json' ? 'JSON' : fileType === 'markdown' ? 'MD' : 'TXT'}
                                                    </text>
                                                </svg>
                                                <div className="absolute -top-1 -right-1 text-white text-xs rounded-full px-1 py-0.5 leading-none font-bold" style={{ fontSize: '8px', backgroundColor: '#6B7280' }}>
                                                    {fileType === 'json' ? 'JSON' : fileType === 'markdown' ? 'MD' : 'TXT'}
                                                </div>
                                            </div>
                                            <div className="flex flex-col">
                                                <span className="text-sm font-medium truncate max-w-[120px]" style={{ color: modeColors.text }}>
                                                    {selectedFile.name}
                                                </span>
                                                <span className="text-xs opacity-70" style={{ color: modeColors.textSecondary }}>
                                                    {fileType === 'json' ? 'JSON Data' : fileType === 'markdown' ? 'Markdown' : 'Text File'} • {(selectedFile.size / 1024).toFixed(1)} KB
                                                </span>
                                            </div>
                                        </div>
                                    )}
                                    {fileType === 'video' && selectedFile && (
                                        <div className="flex items-center gap-2 flex-shrink-0">
                                            {/* Video Thumbnail */}
                                            <div className="relative bg-purple-50 border border-purple-200 rounded-md p-1 flex items-center justify-center" style={{
                                                width: '48px',
                                                height: '48px',
                                                backgroundColor: 'rgb(243 232 255)', // A light purple for video bg
                                                border: `1px solid rgb(233 213 255)` // A slightly darker purple for video border
                                            }}>
                                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-purple-600">
                                                    <polygon points="23 7 16 12 23 17 23 7" />
                                                    <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
                                                </svg>
                                                <div className="absolute inset-0 flex items-center justify-center">
                                                    <div className="bg-purple-600 text-white rounded-full p-1">
                                                        <svg width="8" height="8" viewBox="0 0 24 24" fill="currentColor">
                                                            <polygon points="5 3 19 12 5 21  5 3 19 12 5 21" />
                                                        </svg>
                                                    </div>
                                                </div>
                                                <div className="absolute -top-1 -right-1 text-white text-xs rounded-full px-1 py-0.5 leading-none font-bold" style={{ fontSize: '8px', backgroundColor: '#9333ea' }}> {/* Purple badge */}
                                                    VID
                                                </div>
                                            </div>
                                            <div className="flex flex-col">
                                                <span className="text-sm font-medium truncate max-w-[120px]" style={{ color: modeColors.text }}>
                                                    {selectedFile.name}
                                                </span>
                                                <span className="text-xs opacity-70" style={{ color: modeColors.textSecondary }}>
                                                    Video • {selectedFile.type.split('/')[1]?.toUpperCase()}
                                                </span>
                                            </div>
                                        </div>
                                    )}
                                    {fileType === 'other' && selectedFile && (
                                        <div className="flex items-center gap-2 flex-shrink-0">
                                            {/* Generic File Thumbnail */}
                                            <div className="relative bg-gray-50 border border-gray-200 rounded-md p-1 flex items-center justify-center" style={{
                                                width: '48px',
                                                height: '48px',
                                                backgroundColor: '#F9FAFB', // fileInfo.bgColor
                                                border: `1px solid #E5E7EB` // fileInfo.borderColor
                                            }}>
                                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-gray-600">
                                                    <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
                                                    <polyline points="14 2 14 8 20 8" />
                                                </svg>
                                                <div className="absolute -top-1 -right-1 text-white text-xs rounded-full px-1 py-0.5 leading-none font-bold" style={{ fontSize: '8px', backgroundColor: '#6B7280' }}>
                                                    FILE
                                                </div>
                                            </div>
                                            <div className="flex flex-col">
                                                <span className="text-sm font-medium truncate max-w-[120px]" style={{ color: modeColors.text }}>
                                                    {selectedFile.name}
                                                </span>
                                                <span className="text-xs opacity-70" style={{ color: modeColors.textSecondary }}>
                                                    {selectedFile.type.split('/')[1]?.toUpperCase() || 'FILE'}
                                                </span>
                                            </div>
                                        </div>
                                    )}
                                    <button
                                        onClick={handleRemoveFile}
                                        className="ml-2 bg-red-500 text-white rounded-full p-0.5 text-xs leading-none flex items-center justify-center w-5 h-5 flex-shrink-0"
                                        title="Remove file"
                                    >
                                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                            <line x1="18" y1="6" x2="6" y2="18"></line>
                                            <line x1="6" y1="6" x2="18" y2="18"></line>
                                        </svg>
                                    </button>
                                </div>
                            )}

                            {/* Text input - flex-1 allows it to take remaining space */}
                            <textarea
                                ref={inputRef}
                                className="flex-1 resize-none overflow-hidden h-auto max-h-40 px-4 py-3 text-base bg-transparent focus:outline-none"
                                style={{
                                    color: modeColors.text,
                                }}
                                placeholder="Type a message..."
                                value={question}
                                onChange={(e) => {
                                    setQuestion(e.target.value);
                                    e.target.style.height = 'auto';
                                    e.target.style.height = (e.target.scrollHeight) + 'px';
                                }}
                                onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && !isBotTyping && !isWebSearching && !isLoadingChat && handleSend()} // Disable if any loading state is active
                                disabled={isBotTyping || isWebSearching || isLoadingChat} // Disable if any loading state is active
                                rows={1}
                            />

                            {/* File upload button */}
                            <input
                                type="file"
                                accept="image/*,application/pdf,video/*,.docx,.doc,.pptx,.ppt,.xlsx,.xls,.csv,.txt,.json,.md,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/msword,application/vnd.openxmlformats-officedocument.presentationml.presentation,application/vnd.ms-powerpoint,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel,text/csv,application/csv,text/plain,application/json,text/markdown"
                                ref={fileInputRef}
                                onChange={handleFileUpload}
                                style={{ display: 'none' }}
                                disabled={isBotTyping || isWebSearching || isLoadingChat} // Disable if any loading state is active
                            />
                            <button
                                type="button"
                                onClick={() => !isBotTyping && !isWebSearching && !isLoadingChat && fileInputRef.current?.click()} // Disable if any loading state is active
                                className="p-2 rounded-full transition-colors hover:bg-neutral-700/30 flex-shrink-0 mb-2 self-end"
                                title="Upload File"
                                disabled={isBotTyping || isWebSearching || isLoadingChat} // Disable if any loading state is active
                                style={{ color: currentTheme.primary }}
                            >
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                                    <polyline points="17 8 12 3 7 8"></polyline>
                                    <line x1="12" y1="3" x2="12" y2="15"></line>
                                </svg>
                            </button>

                            {/* Send button */}
                            <button
                                onClick={handleSend}
                                disabled={isBotTyping || isWebSearching || isLoadingChat || (!question.trim() && !selectedFile)} // Disable if any loading state is active or no input
                                className="p-2 mx-1 rounded-full transition text-white disabled:opacity-40 flex-shrink-0 mb-2 self-end"
                                style={{
                                    backgroundColor: currentTheme.primary,
                                }}
                            >
                                {isBotTyping || isWebSearching || isLoadingChat ? (
                                    <svg className="animate-spin" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <circle cx="12" cy="12" r="10" strokeDasharray="30" strokeDashoffset="0"></circle>
                                        <path d="M12 2C6.5 2 2 6.5 2 12"></path>
                                    </svg>
                                ) : (
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <path d="M2 12L20 4L12 22L10 14L2 12Z" fill="currentColor" />
                                    </svg>
                                )}
                            </button>
                        </div>
                    </div>
                </div>
                )}
            </div>
        </div>
    );
}

export default App;