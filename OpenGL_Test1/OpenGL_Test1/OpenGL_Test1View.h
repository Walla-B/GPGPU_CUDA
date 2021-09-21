
// OpenGL_Test1View.h: COpenGLTest1View 클래스의 인터페이스
//
#include<gl/gl.H>
#include<gl/glu.H>
#include<gl/glut.h>
#include<gl/glaux.H>

#pragma comment(lib, "OPENGL32.LIB")
#pragma comment(lib, "GLAUX.LIB")
#pragma comment(lib, "GLU32.LIB")
#pragma comment(lib, "GLUT32.LIB")

#pragma once


class COpenGLTest1View : public CView
{
protected: // serialization에서만 만들어집니다.
	COpenGLTest1View() noexcept;
	DECLARE_DYNCREATE(COpenGLTest1View)

// 특성입니다.
public:
	COpenGLTest1Doc* GetDocument() const;

// 작업입니다.
public:

// 재정의입니다.
public:
	virtual void OnDraw(CDC* pDC);  // 이 뷰를 그리기 위해 재정의되었습니다.
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);

// 구현입니다.
public:
	virtual ~COpenGLTest1View();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// 생성된 메시지 맵 함수
protected:
	afx_msg void OnFilePrintPreview();
	afx_msg void OnRButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnContextMenu(CWnd* pWnd, CPoint point);
	DECLARE_MESSAGE_MAP()
};

#ifndef _DEBUG  // OpenGL_Test1View.cpp의 디버그 버전
inline COpenGLTest1Doc* COpenGLTest1View::GetDocument() const
   { return reinterpret_cast<COpenGLTest1Doc*>(m_pDocument); }
#endif

