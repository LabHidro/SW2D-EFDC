C********************************************************************C
C********************************************************************C
C********************************************************************C
C
      PROGRAM GEFDC
C
C **  PREPROCESS FOR EFDC CODE
C **  CURVILINEAR GRID GENERATION USING
C **  FOR NTYPE=0:   READ IN CELL.INP AND WATER GRID CORNER 
C **                 COORDINATES OF AN ORTHOGONAL GRID  
C **  FOR NTYPE=5:   METHOD OF RYSKIN AND LEAL, 
C **                 J. OF COMP. PHYS. V50, 71-100 (1983),  
C **  FOR NTYPE=1-4: METHOD OF RYSKIN AND LEAL WITH SYMMETRIC
C **                 REFLECTIONS AS SUGGESTED BY CHIKHLIWALA AND 
C **                 YORTSOS, J. OF COMP. PHYS. V57, 391-402 (1985).
C **  FOR NTYPE=6:   THE AREA-ORTHOGONALITY METHOD OF KNUPP,
C **                 J. OF COMP PHYS. V100, 409-418 (1993)
C **  FOR NTYPE=7:   THE QUSICONFORMAL METHOD OF MOBLEY AND STEWART,
C **                 J. OF COMP PHYS. V24, 124-135 (1980)
C **  CARTESIAN COORDINATE SYSTEM PROCESSING
C **  FOR NTYPE=8:   INTERPOLATE DEPTH FILE FOR INPUT CARTESIAN
C **                 GRID IN X,Y (UTM COORDINATES)
C **  FOR NTYPE=9:   CONVERT INPUT LON,LAT CARTESION GRID TO
C **                 VIMS CHES BAY UTM COORDINATES AND INTERPOLATE
C **                 DEPTHS SPECIFIED IN VIMS CHES BAY UTM COORDS.
C 
C **  JOHN M. HAMRICK, VIMS (LAST MODIFIED July 13, 1994)
C **  (DXF OUTPUT CAPABILITIES ADDED BY M. MORTON, TETRA TECH)
C
C********************************************************************C
C
      IMPLICIT REAL*8 (A-H,O-Z)
C
      PARAMETER (IMIND=1,JMIND=1,IMAXD=2000,JMAXD=2000,IGGDIM=2000,
     $           JGGDIM=2000,
     $           NBPD=40000,NGIM=400000,NWCDIM=4000000)
      PARAMETER(NDDMAX=8000000,NVDMAX=4000000)
	  
C
C ches bay parameters
c     PARAMETER (IMIND=1,JMIND=1,IMAXD=100,JMAXD=200,IGGDIM=100,
c    $           JGGDIM=200,
c    $           NBPD=400,NGIM=2000,NWCDIM=20000)
C     PARAMETER(NDDMAX=80000) 
C ches bay parameters
c
      COMMON /ONE/ DEPD(NDDMAX),DEPE(NDDMAX),DEPN(NDDMAX),CDEP,RADM
      COMMON /TWO/ NDEPDAT
      COMMON /VEG/ NVEGDAT,NVEGTYP,ROUGH,NVEGD(NVDMAX),VEGE(NVDMAX),
     $       VEGN(NVDMAX),NNVEG(0:12)
      COMMON /CUTM/  RK,PHI0,CMERID, AE,ECC2,ECC2P
      COMMON /CSMD/ SMD,IB,IE,JB,JE,IJSMD,ISMD,JSMD
      COMMON /CXY/  X(IMIND:IMAXD,JMIND:JMAXD), 
     $              Y(IMIND:IMAXD,JMIND:JMAXD)
C
C********************************************************************C
C
      DIMENSION
     $     XN(IMIND:IMAXD,JMIND:JMAXD),  YN(IMIND:IMAXD,JMIND:JMAXD),
     $     HI(IMIND:IMAXD,JMIND:JMAXD),  HJ(IMIND:IMAXD,JMIND:JMAXD),
     $    RKI(IMIND:IMAXD,JMIND:JMAXD), RKJ(IMIND:IMAXD,JMIND:JMAXD),
     $   RKII(IMIND:IMAXD,JMIND:JMAXD),RKJJ(IMIND:IMAXD,JMIND:JMAXD),
     $   RSGI(IMIND:IMAXD,JMIND:JMAXD), CIJ(IMIND:IMAXD,JMIND:JMAXD),
     $   NVEGIJ(IMIND:IMAXD,JMIND:JMAXD)
      DIMENSION IRED(NGIM),JRED(NGIM),
     $          IBLK(NGIM),JBLK(NGIM)
      DIMENSION KSGI(IMIND:IMAXD,JMIND:JMAXD),
     $          KSBP(IMIND:IMAXD,JMIND:JMAXD),
     $          IJCT(IMIND:IMAXD,JMIND:JMAXD),
     $          XCELL(IMIND:IMAXD,JMIND:JMAXD),
     $          DLONDD(IMIND:IMAXD,JMIND:JMAXD),
     $          YCELL(IMIND:IMAXD,JMIND:JMAXD),
     $          DLATDD(IMIND:IMAXD,JMIND:JMAXD),
     $          DEPFIX(IMIND:IMAXD,JMIND:JMAXD),
     $          DEPCC(IMIND:IMAXD,JMIND:JMAXD)
      DIMENSION IBP(NBPD),JBP(NBPD),XCOMP(NWCDIM),YCOMP(NWCDIM),
     $          ICOMP(NWCDIM),JCOMP(NWCDIM),
     $          IJCTG(IGGDIM,JGGDIM)
C
C **  EXTERNAL FUNCTIONS REQUIRES FOR NTYPE=7, MS METHOD
C
      EXTERNAL FIB,FIE,GJB,GJE
C 
      CHARACTER*80 TITLE,DUMMY
C
 1111 FORMAT(130X)
C
C********************************************************************C
C
      OPEN(2,FILE='depdat.inp',STATUS='UNKNOWN')
      OPEN(3,FILE='CELL.INP',STATUS='UNKNOWN')
      OPEN(4,FILE='gefdc.inp',STATUS='UNKNOWN')
      OPEN(7,FILE='gefdc.out',STATUS='UNKNOWN')
      OPEN(8,FILE='grid.ixy',STATUS='UNKNOWN')
      OPEN(9,FILE='grid.jxy',STATUS='UNKNOWN')
      OPEN(10,FILE='grid.mask',STATUS='UNKNOWN')
      OPEN(12,FILE='grid.cord',STATUS='UNKNOWN')
      OPEN(13,FILE='grid.init',STATUS='UNKNOWN')
      OPEN(14,FILE='DXDY.INP',STATUS='UNKNOWN')
      OPEN(15,FILE='dxdy.diag',STATUS='UNKNOWN')
      OPEN(16,FILE='LXLY.INP',STATUS='UNKNOWN')
      OPEN(66,FILE='gefdc.log',STATUS='UNKNOWN')
      OPEN(67,FILE='depint.log',STATUS='UNKNOWN')
      OPEN(25,file='grid.dxf',STATUS='UNKNOWN')              
      OPEN(26,file='init.dxf',STATUS='UNKNOWN')              
      OPEN(27,file='data.plt',STATUS='UNKNOWN')              
      OPEN(89,file='depspc.out',STATUS='UNKNOWN')              
C     WRITE HEADER FOR DXF FILE (ADDED BY M. MORTON, TETRA TECH)
      WRITE(25,5501) 
      WRITE(26,5501)                                        
5501  FORMAT('  0',/,'SECTION',/,'  2',/,'ENTITIES')         
C
C********************************************************************C
C
C **  READ INPUT DATA AND INITIALIZE
C
C--------------------------------------------------------------------C
C
      READ(4,*)DUMMY
      READ(4,*)DUMMY
      READ(4,*)TITLE
C
      READ(4,*)DUMMY
      READ(4,*)DUMMY
      READ(4,*)NTYPE,NBPP,IMIN,IMAX,JMIN,JMAX,IC,JC
      write(7,*)NTYPE,NBPP,IMIN,IMAX,JMIN,JMAX,IC,JC
C
C     NTYPE = PROBLEM TYPE
C             0, READ IN EXTERNALLY GENERATED GRID
C             1, RL EAST REFLECTION
C             2, RL NORTH REFLECTION
C             3, RL WEST REFLECTION
C             4, RL SOUTH REFLECTION
C             5, RL NO REFLECTION
C             6, AO METHOD
C             7, MS METHOD
C             8, DEPTH INTERPOLATION TO CARTESIAN GRID SPECIFIED  
C                BY cell.inp AND GENERATE dxdy.inp AND lxly.inp 
C                FILES
C             9, DEPTH INTERPOLATION TO CARTESIAN GRID AS FOR 8 
C                CONVERTING INPUT COORDINATE SYSTEM FROM  
C                LONG,LAT TO UTMBAY (VIMS PHYS OCEAN CHES BAY REF)
C     NBPP = NUMBER OF INPUT BOUNDARY POINTS
C     IMIN,IMAX = RANGE OF I GRID INDICES
C     JMIN,JMAX = RANGE OF J GRID INDICES
C     IC = NUMBER OF CELLS IN I DIRECTION
C     JC = NUMBER OF CELLS IN J DIRECTION
C
      READ(4,*)DUMMY
      READ(4,*)DUMMY
      READ(4,*)ISGG,IGM,JGM,DXCG,DYCG,NWTGG
      write(7,*)ISGG,IGM,JGM,DXCG,DYCG,NWTGG
C
C     ISGG = 1, READ IN gcell.inp WHICH DEFINES THE CARTESIAN OR 
C               GRAPHICS GRID OVERLAY
C     IGM    MAXIMUM X OR I CELLS IN CARTESIAN OR GRAPHICS GRID
C     JGM    MAXIMUM Y OF J CELLS IN CARTESIAN OR GRAPHICS GRID
C     DXCG   X GRID SIZE OF CARTESIAN OR GRAPHICS GRID
C     DYCG   Y GRID SIZE OF CARTESIAN OF GRAPHICS GRID
C     NWTGG  NUMBER OF WEIGHTED COMP CELLS USED TO INTERPOLATE 
C            TO THE GRAPHICS GRID
C
      READ(4,*)DUMMY
      READ(4,*)DUMMY
      READ(4,*) CDLON1,CDLON2,CDLON3,CDLAT1,CDLAT2,CDLAT3
      write(7,*) CDLON1,CDLON2,CDLON3,CDLAT1,CDLAT2,CDLAT3
C
C      CDLON1:  6 CONSTANTS TO GIVE CELL CENTER LAT AND LON OR OTHER      
C      CDLON2:    COORDINATES FOR CARTESIAN GRIDS USING THE FORMULAS     
C      CDLON3:    DLON(L)=CDLON1+(CDLON2*FLOAT(I)+CDLON3)/60.            
C      CDLAT1:    DLAT(L)=CDLAT1+(CDLAT2*FLOAT(J)+CDLAT3)/60.            
C      CDLAT2:                                                           
C      CDLAT3:                                                            
C
      READ(4,*)DUMMY
      READ(4,*)DUMMY
      READ(4,*)ITRXM,ITRHM,ITRKM,ITRGM,NDEPSM,NDEPSMF,DEPMIN,DDATADJ
      write(7,*)ITRXM,ITRHM,ITRKM,ITRGM,NDEPSM,NDEPSMF,DEPMIN,DDATADJ
C
C     ITRXM = MAXIMUM NUMBER OF X,Y SOLUTION ITERATIONS
C     ITRHM = MAXIMUM NUMBER OF HI,HJ SOLUTION ITERATIONS
C     ITRKM = MAXIMUM NUMBER OF KJ/KI SOLUTION ITERATIONS
C     ITRGM = MAXIMUM NUMBER OF GRID SOLUTION ITERATIONS
C     NDEPSM = NUMBER SMOOTHING PASSES TO FILL MISSING DEP DAT
C     NDEPSMF = NUMBER FINAL SMOOTHING PASSES AFTER FILL MISSING DEP DAT
C     DEPMIN = MINIMUM DEPTH PASSING DEPDAT.INP DATA
C     DDATADJ = ADJUSTMENT TO DEPTH IN DEPDAT.INP DATA
C
      READ(4,*)DUMMY
      READ(4,*)DUMMY
      READ(4,*)RPX,RPK,RPH,RSQXM,RSQKM,RSQKIM,RSQHM,RSQHIM,RSQHJM
      write(7,*)RPX,RPK,RPH,RSQXM,RSQKM,RSQKIM,RSQHM,RSQHIM,RSQHJM
C
C     RPX,RPK,RPH = RELAXATION PARAMETERS FOR X,Y; KI/KJ; AND HI,HJ
C                   SOR SOLUTIONS
C     RSQXM,RSQKM,RSQHM = MAXIMUM RESIDUAL SQUARED ERROR IN SOR 
C                         SOLUTION FOR X,Y; KJ/KI; AND HI,HJ
C     RSQKIM = CONVERGENCE CRITERIA BASED ON KI/KJ (NOT ACTIVE)
C     RSQHIM = CONVERGENCE CRITERIA BASED ON HI (NOT ACTIVE)
C     RSQHJM = CONVERGENCE CRITERIA BASED ON HJ (NOT ACTIVE)
C
      READ(4,*)DUMMY
      READ(4,*)DUMMY
      READ(4,*)XSHIFT,YSHIFT,HSCALE,RKJDKI,ANGORO
      write(7,*)XSHIFT,YSHIFT,HSCALE,RKJDKI,ANGORO
C
C     XSHIFT,YSHIFT = X,Y COORDINANT SHIFT X,Y=X,Y+XSHIFT,YSHIFT
C     HSCALE = SCALE FACTOR FOR HII AND HJJ WHEN PRINTED TO dxdy.out
C     RKJDKI = ANISOTROPIC STRETCHING OF J COORDINANT (USE 1.)
C     ANGORO = ANGULAR DEVIATION FROM ORTHOGONALITY IN DEG USED
C              AS CONVERGENCE CRITERIA
C
      READ(4,*)DUMMY
      READ(4,*)DUMMY
      READ(4,*)ISIRKI,JSIRKI,ISIHIHJ,JSIHIHJ
      write(7,*)ISIRKI,JSIRKI,ISIHIHJ,JSIHIHJ
C
C     ISIRKI = 1, SOLUTION BASED ON INTERPOLATION OF KJ/KI TO
C              INTERIOR
C     JSIRKI = 1, INTERPOLATE KJ/KI TO INTERIOR WITH CONSTANT 
C              COEFFICIENT DIFFUSION EQUATION
C     ISIHIHJ =1, SOLUTION BASED ON INTERPOLATION OF HI AND HJ TO 
C              INTERIOR, AND THEN DETERMINING KJ/KI=HI/HJ
C     JSIHIHJ = 1, INTERPOLATE HI AND HJ TO INTERIOR WITH CONSTANT 
C              COEFFICIENT DIFFUSION EQUATION
C
      READ(4,*)DUMMY
      READ(4,*)DUMMY
      IF(NTYPE.EQ.7) THEN
        READ(4,*)IB,IE,JB,JE,N7RELAX,NXYIT,ITN7MAX,IJSMD,ISMD,JSMD,
     $           RP7,SERRMAX
      END IF
      IF(IJSMD.NE.0) THEN
        ISMD=0
        JSMD=0
      END IF
      IF(ISMD.NE.0) THEN
        IJSMD=0
        JSMD=0
      END IF
      IF(JSMD.NE.0) THEN
        IJSMD=0
        ISMD=0
      END IF
C
C     IB     = BEGINNING I INDEX MS METHOD
C     IE     = ENDING I INDEX MS METHOD
C     JB     = BEGINNING J INDEX MS METHOD
C     JE     = ENDING J INDEX MS METHOD
C     N7RELAX= MAXIMUM RELAXATION PER INIT LOOP, NTYPE = 7
C     NXYIT  = NUMBER OF ITERS ON EACH X,Y SWEEP, NTYPE = 7 
C     ITN7MAX= MAXIMUM GENERATION ITERS, NTYPE = 7
C     IJSMD  = 1, CALCULATE GLOBAL CONFORMAL MODULE
C     ISMD   = A VALUE IB.LE.ISMD.LE.IE, CALCULATE CONFORMAL
C              MODULE ALONG LINE I=ISMD
C     JSMD   = A VALUE JB.LE.JSMD.LE.JE, CALCULATE CONFORMAL
C              MODULE ALONG LINE J=JSMD
C     RP7    = SOR RELAXATION PARAMETER, NTYPE = 7
C     SERRMAX= MAXIMUM CONFORMAL MODULE ERROR, NTYPE = 7 
C
      READ(4,*)DUMMY
      READ(4,*)DUMMY
      IF(NTYPE.EQ.7) THEN
        READ(4,*)XIBJB,YIBJB
        READ(4,*)XIEJB,YIEJB
        READ(4,*)XIEJE,YIEJE
        READ(4,*)XIBJE,YIBJE
      END IF
C
C     XIBJB,YIBJB = IB,JB COORDINATES
C     XIEJB,YIEJB = IE,JB COORDINATES
C     XIBJE,YIBJE = IB,JE COORDINATES
C     XIEJE,YIEJE = IE,JE COORDINATES
C
      READ(4,*)DUMMY
      READ(4,*)DUMMY
      READ(4,*)ISIDEP,NDEPDAT,CDEP,RADM,ISIDPTYP,SURFELV,ISVEG,
     $         NVEGDAT,NVEGTYP,ROUGH
      write(7,*)ISIDEP,NDEPDAT,CDEP,RADM,ISIDPTYP,SURFELV,ISVEG,
     $         NVEGDAT,NVEGTYP,ROUGH
C
C     ISIDEP   = 1, READ DEPTH FILE AND GENERATE EFDC INPUT FILES
C                DXDY.INP AND LXLY.INP
C     NDEPDAT  = NUMBER OF X, Y, DEPTH FIELDS IN DEPDAT.INP FILE
C     CDEP     = WEIGHTING COEFFICIENT IN DEPTH INTERPOLATION SCHEME
C     RADM     = CONSTANT MULTIPLIER FOR DEPTH INTERPOLATION RADIUS
C     ISIDPTYP = 1, ASSUMES DEPDAT.INP CONTAINS POSITIVE DEPTHS TO
C                TO A BOTTOM BELOW A SEALEVEL DATUM AND THE BOTTOM 
C                ELEVATION IS THE NEGATIVE OF THE DEPTH 
C                2, ASSUMES DEPDAT.INP CONTAINS POSTIVE BOTTOM 
C                ELEVATIONS, LOCAL INITIAL DEPTH IS THEN DETERMINED
C                BY DEPTH=SURFELV-BELB
C                3, ASSUMES DEPDAT.INP CONTAINS POSTIVE BOTTOM 
C                ELEVATIONS WHICH MUST BE CONVERTED TO NEGATIVE VALUES,
C                LOCAL INITIAL DEPTH IS THEN DETERMINED
C                BY DEPTH=SURFELV-BELB
C     SURFELV  = INITIALLY FLAT SURFACE ELEVATION FOR USE WHEN 
C                ISIDPTYP = 2 OR 3.
C     ISVEG    = 1, READ AND INTERPOLATE VEGATION DATA
C     NVEGDAT  = NUMBER OF X,Y,TYPE DATA POINTS
C     NVEGTYP  = NUMBER OF VEGETATION TYPES OR CLASSES    
C
C--------------------------------------------------------------------C
C
      IMAXO=IMAX  
      JMAXO=JMAX
      IMINO=IMIN
      JMINO=JMIN
C
      IF(ISIDEP.EQ.1) THEN
       NCOUNT=0
       DO N=1,NDEPDAT
       NCOUNT=NCOUNT+1
       READ(2,*,ERR=6966)DEPE(N),DEPN(N),DEPTMP
       DEPD(N)=DDATADJ+DEPTMP
       IF(ISIDPTYP.EQ.1) DEPD(N)=MAX(DEPMIN,DEPD(N))
       IF(ISIDPTYP.EQ.3) DEPD(N)=-DEPD(N)
c      IF(NCOUNT.GT.74200) THEN
c         WRITE(6,6969) NCOUNT
c      END IF
       END DO
      WRITE(6,6969)NCOUNT
      END IF
      GO TO 6967
 6966 WRITE(6,6968)
      STOP
 6967 CONTINUE
C
C
      IF(ISVEG.EQ.1) THEN
        OPEN(90,FILE='vegdat.inp',STATUS='UNKNOWN')
        NCOUNT=0
        DO N=1,NVEGDAT
        NCOUNT=NCOUNT+1
        READ(90,*,ERR=6976)VEGE(N),VEGN(N),NVEGD(N)
c      IF(NCOUNT.GT.74200) THEN
c         WRITE(6,6979) NCOUNT
c      END IF
       END DO
       CLOSE(90)
      END IF
      GO TO 6977
 6976 WRITE(6,6978)
      STOP
 6977 CONTINUE
C    
      IF(ISVEG.EQ.1) WRITE(6,6979)NCOUNT
C
 6968 FORMAT('READ ERROR ON FILE depdat.inp AT NCOUNT =',I10//)
 6969 FORMAT(I10,' DEPTH DATA POINTS READ IN',//)
 6978 FORMAT('READ ERROR ON FILE vegdat.inp AT NCOUNT =',I10//)
 6979 FORMAT(I10,' VEGATATION DATA POINTS READ IN',//)
C
C********************************************************************C
C
      IF(NTYPE.GE.5.AND.NTYPE.LE.6)THEN
       NBP=NBPP
      ELSE
       NBP=2*NBPP-2
      END IF
C
      IF (NTYPE.EQ.1) THEN
       IMAX=2*IMAX-IMIN
      END IF
      IF (NTYPE.EQ.2) THEN
        JMAX=2*JMAX-JMIN
      END IF
      IF (NTYPE.EQ.3) THEN
       IMIN=2*IMIN-IMAX
      END IF
      IF (NTYPE.EQ.4) THEN
       JMIN=2*JMIN-JMAX
      END IF
C
C--------------------------------------------------------------------C
C
      DO J=JMIN,JMAX
      DO I=IMIN,IMAX
      X(I,J)=0.
      Y(I,J)=0.
      XN(I,J)=0.
      YN(I,J)=0.
      HI(I,J)=1.
      HJ(I,J)=1.
      RKI(I,J)=1.
      RKII(I,J)=1.
      RKJ(I,J)=RKJDKI
      RKJJ(I,J)=RKJDKI
      KSBP(I,J)=0
      KSGI(I,J)=0
      RSGI(I,J)=0.
      DEPCC(I,J)=0.
      DEPFIX(I,J)=0.
      NVEGIJ(I,J)=0
      END DO
      END DO
C
      IF(NTYPE.GE.8) GO TO 8000
C
C--------------------------------------------------------------------C
C
C **  NTYPE = 1 REFLECTION TO EAST OF CONSTANT I LINE
C **  NTYPE = 3 REFLECTION TO WEST OF CONSTANT I LINE
C
      IF (NTYPE.EQ.1.OR.NTYPE.EQ.3) THEN
C
C **  READ IN LAST POINT COORDINATES
C
      READ(4,*)DUMMY
      READ(4,*)DUMMY
C
      READ(4,*)ILT,JLT,X(ILT,JLT),Y(ILT,JLT)
      XL=X(ILT,JLT)
      YL=Y(ILT,JLT)
C
C **  READ IN FIRST POINT COORDINATES
C
      READ(4,*)DUMMY
      READ(4,*)DUMMY
C
      READ(4,*)IFT,JFT,X(IFT,JFT),Y(IFT,JFT)
      WRITE(10,101)X(IFT,JFT),Y(IFT,JFT)
      XF=X(IFT,JFT)
      YF=Y(IFT,JFT)
      IBP(1)=IFT
      JBP(1)=JFT
      KSBP(IFT,JFT)=1
      KSGI(IFT,JFT)=1
      RSGI(IFT,JFT)=1.
C
C **  CALCULATE COORDINATE ROTATION FOR REFLECTION
C 
      DX=XF-XL
      DY=YF-YL
      ANG=ATAN2(DY,DX)
C
      WRITE(6,6001)ANG
      WRITE(66,6001)ANG
 6001 FORMAT(1X,'ANG = ',E11.4)
C
C **  READ INTERMEDIATE BOUNDARY POINT COORDINATES
C
      DO N=2,NBPP-1
      NN=N+NBPP-1
      READ(4,*)I,J,X(I,J),Y(I,J)
      WRITE(10,101)X(I,J),Y(I,J)
      XN(I,J)=(X(I,J)-XL)*COS(ANG)+(Y(I,J)-YL)*SIN(ANG)
      YN(I,J)=-(X(I,J)-XL)*SIN(ANG)+(Y(I,J)-YL)*COS(ANG)
      IBP(N)=I
      JBP(N)=J
      KSBP(I,J)=1
      KSGI(I,J)=1
      RSGI(I,J)=1.
      II=2*IFT-I
      IBP(NN)=II
      JBP(NN)=J
      KSBP(II,J)=1
      KSGI(II,J)=1
      RSGI(II,J)=1.
      XN(II,J)=XN(I,J)
      YN(II,J)=-YN(I,J)
      X(I,J)=XL+XN(I,J)*COS(ANG)-YN(I,J)*SIN(ANG)
      Y(I,J)=YL+XN(I,J)*SIN(ANG)+YN(I,J)*COS(ANG)
      X(II,J)=XL+XN(II,J)*COS(ANG)-YN(II,J)*SIN(ANG)
      Y(II,J)=YL+XN(II,J)*SIN(ANG)+YN(II,J)*COS(ANG)
      END DO
C
      READ(4,*)I,J,X(I,J),Y(I,J)
      WRITE(10,101)X(I,J),Y(I,J)
      WRITE(10,101)XF,YF
      IBP(NBPP)=I
      JBP(NBPP)=J
      KSBP(I,J)=1
      KSGI(I,J)=1
      RSGI(I,J)=1.
C
      END IF
C
C--------------------------------------------------------------------C
C
C **  NTYPE = 2 REFLECTION TO NORTH OF CONSTANT J LINE
C **  NTYPE = 4 REFLECTION TO SOUTH OF CONSTANT J LINE
C
      IF (NTYPE.EQ.2.OR.NTYPE.EQ.4) THEN
C
C **  READ IN LAST POINT COORDINATES
C
      READ(4,*)DUMMY
      READ(4,*)DUMMY
C
      READ(4,*)ILT,JLT,X(ILT,JLT),Y(ILT,JLT)
      XL=X(ILT,JLT)
      YL=Y(ILT,JLT)
C
C **  READ IN FIRST POINT COORDINATES
C
      READ(4,*)DUMMY
      READ(4,*)DUMMY
C
      READ(4,*)IFT,JFT,X(IFT,JFT),Y(IFT,JFT)
      WRITE(10,101)X(IFT,JFT),Y(IFT,JFT)
      XF=X(IFT,JFT)
      YF=Y(IFT,JFT)
      IBP(1)=IFT
      JBP(1)=JFT
      KSBP(IFT,JFT)=1
      KSGI(IFT,JFT)=1
      RSGI(IFT,JFT)=1.
C
C **  CALCULATE COORDINATE ROTATION FOR REFLECTION
C 
      DX=XF-XL
      DY=YF-YL
      ANG=ATAN2(DY,DX)
C
C **  READ INTERMEDIATE BOUNDARY POINT COORDINATES
C
      DO N=2,NBPP-1
      NN=N+NBPP-1
      READ(4,*)I,J,X(I,J),Y(I,J)
      WRITE(10,101)X(I,J),Y(I,J)
      XN(I,J)=(X(I,J)-XL)*COS(ANG)+(Y(I,J)-YL)*SIN(ANG)
      YN(I,J)=-(X(I,J)-XL)*SIN(ANG)+(Y(I,J)-YL)*COS(ANG)
      IBP(N)=I
      JBP(N)=J
      KSBP(I,J)=1
      KSGI(I,J)=1
      RSGI(I,J)=1.
      JJ=2*JFT-J
      IBP(NN)=I
      JBP(NN)=JJ
      KSBP(I,JJ)=1
      KSGI(I,JJ)=1
      RSGI(I,JJ)=1.
      XN(I,JJ)=XN(I,J)
      YN(I,JJ)=-YN(I,J)
      X(I,J)=XL+XN(I,J)*COS(ANG)-YN(I,J)*SIN(ANG)
      Y(I,J)=YL+XN(I,J)*SIN(ANG)+YN(I,J)*COS(ANG)
      X(I,JJ)=XL+XN(I,JJ)*COS(ANG)-YN(I,JJ)*SIN(ANG)
      Y(I,JJ)=YL+XN(I,JJ)*SIN(ANG)+YN(I,JJ)*COS(ANG)
      END DO
C
      READ(4,*)I,J,X(I,J),Y(I,J)
      WRITE(10,101)X(I,J),Y(I,J)
      WRITE(10,101)XF,YF
      IBP(NBPP)=I
      JBP(NBPP)=J
      KSBP(I,J)=1
      KSGI(I,J)=1
      RSGI(I,J)=1.
C
      END IF
C
C--------------------------------------------------------------------C
C
C **  NTYPE = 5 OR 6, NO REFLECTION, BOUNDARY POINTS CLOSE
C
      IF(NTYPE.GE.5.AND.NTYPE.LE.6) THEN
C
C **  READ IN LAST POINT COORDINATES
C
      READ(4,*)DUMMY
      READ(4,*)DUMMY
C
      READ(4,*)ILT,JLT,X(ILT,JLT),Y(ILT,JLT)
      write(7,*)ILT,JLT,X(ILT,JLT),Y(ILT,JLT)
      XL=X(ILT,JLT)
      YL=Y(ILT,JLT)
C
C **  READ IN FIRST POINT COORDINATES
C
      READ(4,*)DUMMY
      READ(4,*)DUMMY
C
      READ(4,*)IFT,JFT,X(IFT,JFT),Y(IFT,JFT)
      write(7,*)IFT,JFT,X(IFT,JFT),Y(IFT,JFT)
      WRITE(10,101)X(IFT,JFT),Y(IFT,JFT)
      XF=X(IFT,JFT)
      YF=Y(IFT,JFT)
      XN(IFT,JFT)=0.
      YN(IFT,JFT)=0.
      IBP(1)=IFT
      JBP(1)=JFT
      KSBP(IFT,JFT)=1
      KSGI(IFT,JFT)=1
      RSGI(IFT,JFT)=1.
C
C **  READ INTERMEDIATE BOUNDARY POINT COORDINATES
C
      DO N=2,NBPP-1
      READ(4,*)I,J,X(I,J),Y(I,J)
      write(7,*)I,J,X(I,J),Y(I,J)
      WRITE(10,101)X(I,J),Y(I,J)
      IBP(N)=I
      JBP(N)=J
      KSBP(I,J)=1
      KSGI(I,J)=1
      RSGI(I,J)=1.
      END DO
C
      READ(4,*)I,J,X(I,J),Y(I,J)
      write(7,*)I,J,X(I,J),Y(I,J)
      WRITE(10,101)X(I,J),Y(I,J)
      WRITE(10,101)XF,YF
      IBP(NBPP)=I
      JBP(NBPP)=J
      KSBP(I,J)=1
      KSGI(I,J)=1
      RSGI(I,J)=1.
C
      END IF
C
  101 FORMAT(1X,E12.4,5X,E12.4)
C
C********************************************************************C
C
C **  SHIFT COORDINATES
C
C--------------------------------------------------------------------C
C
      DO J=JMIN,JMAX
      DO I=IMIN,IMAX
      X(I,J)=X(I,J)+XSHIFT
      Y(I,J)=Y(I,J)+YSHIFT
      END DO
      END DO
C
C********************************************************************C
C
C **  READ CELL TYPES FROM FILE cell.inp
C
C--------------------------------------------------------------------C
C
C **  SKIP OVER TITLE AND AND HEADER LINES AND DETERMINE FILE FORMAT
C 
      DO IS=1,4
      READ(3,1111)
      END DO
      READ(3,3003)JCTMP
      CLOSE(3)
      OPEN (3,FILE='CELL.INP',STATUS='UNKNOWN')
C
C **  SKIP OVER TITLE AND AND HEADER LINES
C 
      DO IS=1,4
      READ(3,1111)
      END DO
C
      IF(JCTMP.NE.JC) THEN
C **    READ OLD FILE FORMAT
        DO JT=1,JC,120
        JF=JT
        JLAST=JT+119
        IF(JLAST.GT.JC) JLAST=JC
        WRITE (7,700)JF,JLAST 
        DO I=1,IC
        READ (3,3001) (IJCT(I,J),J=JF,JLAST)
        WRITE(7,701)I,(IJCT(I,J),J=JF,JLAST)     
        END DO
        END DO
       ELSE
C **    READ NEW FILE FORMAT
C       READ(3,3003)JCTMP
        DO IT=1,IC,120
        IFIRST=IT
        ILAST=IT+119
        IF(ILAST.GT.IC) ILAST=IC
        WRITE (7,7001)IFIRST,ILAST 
        DO J=JC,1,-1
        READ (3,3003)JDUMY,(IJCT(I,J),I=IFIRST,ILAST)
        WRITE (7,701)JDUMY,(IJCT(I,J),I=IFIRST,ILAST)
        END DO
        END DO
      END IF
C
 3001 FORMAT(120I1)
 3002 FORMAT(2I5)
 3003 FORMAT(I3,2X,125I1)
  700 FORMAT(1X,'  CELL TYPE ARRAY,J=',I5,2X,'TO J=',I5,//)
  701 FORMAT(1X,I3,2X,125I1)
  707 FORMAT(1X,'  GCELL TYPE ARRAY,J=',I5,2X,'TO J=',I5,//)
 7001 FORMAT(1X,'  CELL TYPE ARRAY,I=',I5,2X,'TO I=',I5,//)
 7007 FORMAT(1X,'  GELL TYPE ARRAY,I=',I5,2X,'TO I=',I5,//)
C
C********************************************************************C
C
      IF(NTYPE.EQ.7) GO TO 7000
C
C********************************************************************C
C
C     NTYPE=0, READ IN EXTERNALLY GENERTATED GRID CONSISTENT WITH
C     cell.inp FROM gridext.inp
C
      IF(NTYPE.EQ.0) THEN
C
      OPEN (1,FILE='gridext.inp',STATUS='UNKNOWN')
C
 5998 CONTINUE
      READ(1,*,END=5999)I,J,X(I,J),Y(I,J)
      X(I,J)=X(I,J)+XSHIFT
      Y(I,J)=Y(I,J)+YSHIFT
      GO TO 5998
 5999 CONTINUE
C
      CLOSE(1)
C
      GO TO 3500      
C
      END IF
C
C********************************************************************C
C
C **  GENERATE INTERIOR-EXTERIOR ARRAY
C
C--------------------------------------------------------------------C
C
      DO J=JMINO+1,JMAXO
      DO I=IMINO+1,IMAXO
      IF(KSGI(I,J).EQ.0)THEN
       IF(IJCT(I,J).GE.1.AND.IJCT(I,J).LE.5)THEN
        KSGI(I,J)=1
        RSGI(I,J)=1.
       END IF
       IF(IJCT(I-1,J).GE.1.AND.IJCT(I-1,J).LE.5)THEN
        KSGI(I,J)=1
        RSGI(I,J)=1.
       END IF
       IF(IJCT(I,J-1).GE.1.AND.IJCT(I,J-1).LE.5)THEN
        KSGI(I,J)=1
        RSGI(I,J)=1.
       END IF
      END IF
      END DO
      END DO
C
      IF (NTYPE.EQ.1) THEN
       DO I=IFT-1,IMIN,-1
       IR=2*IFT-I
       DO J=JMIN,JMAX
       KSGI(IR,J)=KSGI(I,J)
       RSGI(IR,J)=RSGI(I,J)
       END DO
       END DO
      END IF
C
      IF (NTYPE.EQ.3) THEN
       DO I=IFT+1,IMAX
       IR=2*IFT-I
       DO J=JMIN,JMAX
       KSGI(IR,J)=KSGI(I,J)
       RSGI(IR,J)=RSGI(I,J)
       END DO
       END DO
      END IF
C
      IF (NTYPE.EQ.2) THEN
       DO I=IMIN,IMAX
       DO J=JFT-1,JMIN,-1
       JR=2*JFT-J
       KSGI(I,JR)=KSGI(I,J)
       RSGI(I,JR)=RSGI(I,J)
       END DO
       END DO
      END IF
C
      IF (NTYPE.EQ.4) THEN
       DO I=IMIN,IMAX
       DO J=JFT+1,JMAX
       JR=2*JFT-J
       KSGI(I,JR)=KSGI(I,J)
       RSGI(I,JR)=RSGI(I,J)
       END DO
       END DO
      END IF
C
      WRITE(7,702)
C
      DO JT=JMIN,JMAX,120
      JF=JT
      JLAST=JT+119
      IF(JLAST.GT.JMAX) JLAST=JMAX
      WRITE (7,700)JF,JLAST 
      DO I=IMIN,IMAX 
      WRITE(7,703)I,(KSGI(I,J),J=JF,JLAST)
      END DO
      END DO
C
  702 FORMAT(1X,'KSGI ARRAY',//)
  703 FORMAT(1X,I3,2X,120I1)
C
C********************************************************************C
C
C **  SET INTERIOR RED AND BLACK CELL SEQUENCES FOR RB SOR SOLUTIONS
C **  FOR NTYPE = 1-6
C
C--------------------------------------------------------------------C
C
      NRED=0
      NBLK=0
C
      DO I=IMIN,IMAX
      DO J=JMIN,JMAX
C
      IF(KSBP(I,J).EQ.0.AND.KSGI(I,J).EQ.1)THEN
      IPJ=I+J
      IR=MOD(IPJ,2)
      IF(IR.EQ.0)THEN
       NRED=NRED+1
       IRED(NRED)=I
       JRED(NRED)=J
      ELSE
       NBLK=NBLK+1
       IBLK(NBLK)=I
       JBLK(NBLK)=J
      END IF   
      END IF
C
      END DO
      END DO   
C
C********************************************************************C
C
C **  INITIALIZE X AND Y TO INTERIOR BY CONSTANT COEFFICIENT 
C **  LAPLACE EQUATION RELAXATION, RKII AND RKJJ HAVE BEEN 
C **  INITIALIZED TO 1.0
C
C--------------------------------------------------------------------C
C
      DO J=JMIN,JMAX
      DO I=IMIN,IMAX
      CIJ(I,J)=-(RKII(I,J)+RKII(I-1,J)+RKJJ(I,J)+RKJJ(I,J-1))
      END DO
      END DO
C
      ITR=0
C
  200 CONTINUE
      ITR=ITR+1
      RSQX=0.
      RSQY=0.
C
      DO N=1,NRED
      I=IRED(N)
      J=JRED(N)
      RSD=RKII(I,J)*X(I+1,J)+RKII(I-1,J)*X(I-1,J)
     $    +RKJJ(I,J)*X(I,J+1)+RKJJ(I,J-1)*X(I,J-1)
     $    +CIJ(I,J)*X(I,J)
      X(I,J)=X(I,J)-RPX*RSD/CIJ(I,J)
      RSQX=RSQX+RSD*RSD
      RSD=RKII(I,J)*Y(I+1,J)+RKII(I-1,J)*Y(I-1,J)
     $    +RKJJ(I,J)*Y(I,J+1)+RKJJ(I,J-1)*Y(I,J-1)
     $    +CIJ(I,J)*Y(I,J)
      Y(I,J)=Y(I,J)-RPX*RSD/CIJ(I,J)
      RSQY=RSQY+RSD*RSD
      END DO
C
      DO N=1,NBLK
      I=IBLK(N)
      J=JBLK(N)
      RSD=RKII(I,J)*X(I+1,J)+RKII(I-1,J)*X(I-1,J)
     $    +RKJJ(I,J)*X(I,J+1)+RKJJ(I,J-1)*X(I,J-1)
     $    +CIJ(I,J)*X(I,J)
      X(I,J)=X(I,J)-RPX*RSD/CIJ(I,J)
      RSQX=RSQX+RSD*RSD
      RSD=RKII(I,J)*Y(I+1,J)+RKII(I-1,J)*Y(I-1,J)
     $    +RKJJ(I,J)*Y(I,J+1)+RKJJ(I,J-1)*Y(I,J-1)
     $    +CIJ(I,J)*Y(I,J)
      Y(I,J)=Y(I,J)-RPX*RSD/CIJ(I,J)
      RSQY=RSQY+RSD*RSD
      END DO
C
      IF(ITR.GE.ITRXM)GO TO 201
      IF(RSQX.GE.RSQXM.OR.RSQY.GE.RSQXM)GO TO 200
C
  201 CONTINUE
C
      WRITE(6,601)ITR,RSQX,RSQY
      WRITE(66,601)ITR,RSQX,RSQY
  601 FORMAT(1X,'DIFF INITIAL X&Y, ITER = ',I3,
     $          ' RSX,RSY =',2(2X,E11.4))
C
C--------------------------------------------------------------------C
C
C **  OUTPUT INITIAL GRID FOR PLOTTING
C
      IF(NTYPE.EQ.1)THEN
       DO J=JMIN,JMAX
       WRITE(9,92)J
       WRITE(13,92)J
       DO I=IMIN,IMAXO
       IF(KSGI(I,J).EQ.1)THEN
C       WRITE(7,77)I,J,X(I,J),Y(I,J)
        WRITE(9,90)X(I,J),Y(I,J)
        WRITE(13,90)X(I,J),Y(I,J)
       END IF
       END DO
       END DO
       DO I=IMIN,IMAXO
       WRITE(9,91)I
       WRITE(13,91)I
       DO J=JMIN,JMAX
       IF(KSGI(I,J).EQ.1)THEN
        WRITE(8,90)X(I,J),Y(I,J)
        WRITE(13,90)X(I,J),Y(I,J)
       END IF
       END DO
       END DO
      END IF
C
      IF(NTYPE.EQ.2)THEN
       DO J=JMIN,JMAXO
       WRITE(9,92)J
       WRITE(13,92)J
       DO I=IMIN,IMAX
       IF(KSGI(I,J).EQ.1)THEN
C       WRITE(7,77)I,J,X(I,J),Y(I,J)
        WRITE(9,90)X(I,J),Y(I,J)
        WRITE(13,90)X(I,J),Y(I,J)
       END IF
       END DO
       END DO
       DO I=IMIN,IMAX
       WRITE(9,91)I
       WRITE(13,91)I
       DO J=JMIN,JMAXO
       IF(KSGI(I,J).EQ.1)THEN
        WRITE(8,90)X(I,J),Y(I,J)
        WRITE(13,90)X(I,J),Y(I,J)
       END IF
       END DO
       END DO
      END IF
C           
      IF(NTYPE.EQ.3)THEN
       DO J=JMIN,JMAX
       WRITE(9,92)J
       WRITE(13,92)J
       DO I=IMINO,IMAX
       IF(KSGI(I,J).EQ.1)THEN
C       WRITE(7,77)I,J,X(I,J),Y(I,J)
        WRITE(9,90)X(I,J),Y(I,J)
        WRITE(13,90)X(I,J),Y(I,J)
       END IF
       END DO
       END DO
       DO I=IMINO,IMAX
       WRITE(9,91)I
       WRITE(13,91)I
       DO J=JMIN,JMAX
       IF(KSGI(I,J).EQ.1)THEN
        WRITE(8,90)X(I,J),Y(I,J)
        WRITE(13,90)X(I,J),Y(I,J)
       END IF
       END DO
       END DO
      END IF
C           
      IF(NTYPE.EQ.4)THEN
       DO J=JMINO,JMAX
       WRITE(9,92)J
       WRITE(13,92)J
       DO I=IMIN,IMAX
       IF(KSGI(I,J).EQ.1)THEN
C       WRITE(7,77)I,J,X(I,J),Y(I,J)
        WRITE(8,90)X(I,J),Y(I,J)
        WRITE(13,90)X(I,J),Y(I,J)
       END IF
       END DO
       END DO
       DO I=IMIN,IMAX
       WRITE(9,91)I
       WRITE(13,91)I
       DO J=JMINO,JMAX
       IF(KSGI(I,J).EQ.1)THEN
        WRITE(8,90)X(I,J),Y(I,J)
        WRITE(13,90)X(I,J),Y(I,J)
       END IF
       END DO
       END DO
      END IF
C
      IF(NTYPE.GE.5)THEN
       DO J=JMIN,JMAX
       WRITE(9,92)J
       WRITE(13,92)J
       DO I=IMIN,IMAX
       IF(KSGI(I,J).EQ.1)THEN
C       WRITE(7,77)I,J,X(I,J),Y(I,J)
        WRITE(9,90)X(I,J),Y(I,J)
        WRITE(13,90)X(I,J),Y(I,J)
       END IF
       END DO
       END DO
       DO I=IMIN,IMAX
       WRITE(9,91)I
       WRITE(13,91)I
       DO J=JMIN,JMAX
       IF(KSGI(I,J).EQ.1)THEN
        WRITE(8,90)X(I,J),Y(I,J)
        WRITE(13,90)X(I,J),Y(I,J)
       END IF
       END DO
       END DO
      END IF
C           
C--------------------------------------------------------------------C
C
C **  OUTPUT INITIAL X AND Y FIELDS
C
      WRITE(7,75)
      WRITE(7,76)
C
      DO J=JMIN,JMAX
      DO I=IMIN,IMAX
      IF(KSGI(I,J).EQ.1)THEN
       WRITE(7,77)I,J,X(I,J),Y(I,J)
      END IF
      END DO
      END DO
C
      WRITE(7,78)
C
   75 FORMAT(1X,'INITIAL X AND Y FIELDS',/)
   76 FORMAT(5X,'I',10X,'J',11X,'X',12X,'Y',/)
   77 FORMAT(1X,I10,5X,I10,5X,F12.4,5X,F12.4)
   78 FORMAT(//////)
   79 FORMAT(1X,'FINAL X AND Y FIELDS',/)
C
C--------------------------------------------------------------------C
C
C **  WRITE DXY FILE OF INITIAL GRID
C
C **  BEGIN DXF WRITE SEQUENCE TO FILE init.dxf
C       The following section added by M.R. Morton, Tetra Tech:
C       Write coordinates of each cell corner to DXF format file.
C       Each cell will be a closed polyline in the DXF file:
C       IJCT = 1, triangular computational cell in NE quadrant:
      DO J=JMIN,JMAX
      DO I=IMIN,IMAX
        if (IJCT(I,J) .eq. 1) then                     
          write(26,1650)                               
          write(26,1651) X(I  ,J  ),Y(I  ,J  )         
          write(26,1651) X(I+1,J  ),Y(I+1,J  )         
          write(26,1651) X(I  ,J+1),Y(I  ,J+1)         
          write(26,1651) X(I  ,J  ),Y(I  ,J  )         
          write(26,1652)                               
        end if                                         
C       IJCT = 4, computational cell in SE quadrant:       
        if (IJCT(I,J) .eq. 4) then                     
          write(26,1650)                               
          write(26,1651) X(I  ,J  ),Y(I  ,J  )         
          write(26,1651) X(I+1,J  ),Y(I+1,J  )         
          write(26,1651) X(I+1,J+1),Y(I+1,J+1)         
          write(26,1651) X(I  ,J  ),Y(I  ,J  )         
          write(26,1652)                               
        end if                                         
C       IJCT = 3, computational cell in SW quadrant:       
        if (IJCT(I,J) .eq. 3) then                     
          write(26,1650)                               
          write(26,1651) X(I+1,J  ),Y(I+1,J  )         
          write(26,1651) X(I+1,J+1),Y(I+1,J+1)         
          write(26,1651) X(I  ,J+1),Y(I  ,J+1)         
          write(26,1651) X(I+1,J  ),Y(I+1,J  )         
          write(26,1652)                               
        end if                                         
C       IJCT = 2, computational cell in NW quadrant:       
        if (IJCT(I,J) .eq. 2) then                     
          write(26,1650)                               
          write(26,1651) X(I  ,J  ),Y(I  ,J  )         
          write(26,1651) X(I+1,J+1),Y(I+1,J+1)         
          write(26,1651) X(I  ,J+1),Y(I  ,J+1)         
          write(26,1651) X(I  ,J  ),Y(I  ,J  )         
          write(26,1652)                               
        end if                                         
C       IJCT = 5, normal four-sided water cell:            
        if (IJCT(I,J) .eq. 5) then                     
          write(26,1650)                               
          write(26,1651) X(I  ,J  ),Y(I  ,J  )         
          write(26,1651) X(I+1,J  ),Y(I+1,J  )         
          write(26,1651) X(I+1,J+1),Y(I+1,J+1)         
          write(26,1651) X(I  ,J+1),Y(I  ,J+1)         
          write(26,1651) X(I  ,J  ),Y(I  ,J  )         
          write(26,1652)                               
        end if
      END DO
      END DO
C **  END DXF WRITE SEQUENCE TO FILE grid.dxf   
C                      
C********************************************************************C
C
C **  NTYPE = (1-5) GRID GENERATION LOOP 
C
C--------------------------------------------------------------------C
C
      ITRG=0
      IF (NTYPE.EQ.6) GO TO 2000
C
 1000 CONTINUE
      ITRG=ITRG+1    
C
C--------------------------------------------------------------------C
C
C **  DETERMINE METRIC COEFFIEIENTS ON THE BOUNDARIES
C
      DO N=1,NBP
      I=IBP(N)
      J=JBP(N)
C
      IF(KSGI(I-1,J).EQ.0.AND.KSGI(I+1,J).EQ.1)THEN
       IF(KSGI(I+2,J).EQ.0)THEN
        DXDI=X(I+1,J)-X(I,J)
        DYDI=Y(I+1,J)-Y(I,J)
       ELSE
        DXDI=0.5*(4.*X(I+1,J)-3.*X(I,J)-X(I+2,J))
        DYDI=0.5*(4.*Y(I+1,J)-3.*Y(I,J)-Y(I+2,J))
       END IF
      END IF
C
      IF(KSGI(I+1,J).EQ.0.AND.KSGI(I-1,J).EQ.1)THEN
       IF(KSGI(I-2,J).EQ.0)THEN
        DXDI=X(I,J)-X(I-1,J)
        DYDI=Y(I,J)-Y(I-1,J)
       ELSE
        DXDI=0.5*(3.*X(I,J)-4.*X(I-1,J)+X(I-2,J))
        DYDI=0.5*(3.*Y(I,J)-4.*Y(I-1,J)+Y(I-2,J))
       END IF
      END IF
C
      IF(KSGI(I+1,J).EQ.1.AND.KSGI(I-1,J).EQ.1)THEN
       DXDI=0.5*(X(I+1,J)-X(I-1,J))
       DYDI=0.5*(Y(I+1,J)-Y(I-1,J))
      END IF
C
      IF(KSGI(I,J-1).EQ.0.AND. KSGI(I,J+1).EQ.1)THEN
       IF(KSGI(I,J+2).EQ.0)THEN
        DXDJ=X(I,J+1)-X(I,J)
        DYDJ=Y(I,J+1)-Y(I,J)
       ELSE
        DXDJ=0.5*(4.*X(I,J+1)-3.*X(I,J)-X(I,J+2))
        DYDJ=0.5*(4.*Y(I,J+1)-3.*Y(I,J)-Y(I,J+2))
       END IF
      END IF
C
      IF(KSGI(I,J+1).EQ.0.AND.KSGI(I,J-1).EQ.1)THEN
       IF(KSGI(I,J-2).EQ.0)THEN
        DXDJ=X(I,J)-X(I,J-1)
        DYDJ=Y(I,J)-Y(I,J-1)
       ELSE
        DXDJ=0.5*(3.*X(I,J)-4.*X(I,J-1)+X(I,J-2))
        DYDJ=0.5*(3.*Y(I,J)-4.*Y(I,J-1)+Y(I,J-2))
       END IF
      END IF
C
      IF(KSGI(I,J+1).EQ.1.AND.KSGI(I,J-1).EQ.1)THEN
       DXDJ=0.5*(X(I,J+1)-X(I,J-1))
       DYDJ=0.5*(Y(I,J+1)-Y(I,J-1))
      END IF
C
      HI(I,J)=SQRT(DXDI*DXDI+DYDI*DYDI)
      HJ(I,J)=SQRT(DXDJ*DXDJ+DYDJ*DYDJ)
      RKI(I,J)=HJ(I,J)/HI(I,J)
C
      END DO
C
C--------------------------------------------------------------------C
C
C **  INTERPOLATE RKI=HJ/HI TO INTERIOR BY LAPLACE EQ RELAXATION
C
      IF (ISIRKI.EQ.1) THEN
C
      IF (JSIRKI.EQ.1) THEN
       DO J=JMIN,JMAX
       DO I=IMIN,IMAX
       RKII(I,J)=1.
       RKJJ(I,J)=RKJDKI
       END DO
       END DO
      END IF
C
      ITR=0
C
      DO J=JMIN,JMAX
      DO I=IMIN,IMAX
      CIJ(I,J)=-(RKII(I,J)+RKII(I-1,J)+RKJJ(I,J)+RKJJ(I,J-1))
      END DO
      END DO
C
  300 CONTINUE
      ITR=ITR+1
      RSQ=0.
C
      DO N=1,NRED
      I=IRED(N)
      J=JRED(N)
      RSD=RKII(I,J)*RKI(I+1,J)+RKII(I-1,J)*RKI(I-1,J)
     $   +RKJJ(I,J)*RKI(I,J+1)+RKJJ(I,J-1)*RKI(I,J-1)
     $   +CIJ(I,J)*RKI(I,J)
      RKI(I,J)=RKI(I,J)-RPK*RSD/CIJ(I,J)
      RSQ=RSQ+RSD*RSD
      END DO
C
      DO N=1,NBLK
      I=IBLK(N)
      J=JBLK(N)
      RSD=RKII(I,J)*RKI(I+1,J)+RKII(I-1,J)*RKI(I-1,J)
     $   +RKJJ(I,J)*RKI(I,J+1)+RKJJ(I,J-1)*RKI(I,J-1)
     $   +CIJ(I,J)*RKI(I,J)
      RKI(I,J)=RKI(I,J)-RPK*RSD/CIJ(I,J)
      RSQ=RSQ+RSD*RSD
      END DO
C
      IF (ITR.GE.ITRKM) GO TO 301
      IF (RSQ.GE.RSQKM) GO TO 300
C
  301 CONTINUE
      WRITE(6,602)ITR,RSQ
      WRITE(66,602)ITR,RSQ
C
      DO J=JMIN,JMAX
      DO I=IMIN,IMAX
      RKJ(I,J)=RKJDKI/RKI(I,J)
      END DO
      END DO
C
      DO J=JMIN,JMAX
      DO I=IMIN,IMAX-1
      RKII(I,J)=0.5*(RKI(I+1,J)+RKI(I,J))
      END DO
      END DO
C
      DO J=JMIN,JMAX-1
      DO I=IMIN,IMAX
      RKJJ(I,J)=0.5*(RKJ(I,J+1)+RKJ(I,J))
      END DO
      END DO
C
      END IF
C
  602 FORMAT(1X,'DIFFUSE RKI, ITERATION = ',I3,' RSK =',2X,E11.4)
C
C--------------------------------------------------------------------C
C
C **  INTERPOLATE HI AND HJ 
C
      IF (ISIHIHJ.EQ.1) THEN
C
      IF (JSIHIHJ.EQ.1) THEN
       DO J=JMIN,JMAX
       DO I=IMIN,IMAX
       RKII(I,J)=1.
       RKJJ(I,J)=RKJDKI
       END DO
       END DO
      END IF
C
      DO J=JMIN,JMAX
      DO I=IMIN,IMAX
      CIJ(I,J)=-(RKII(I,J)+RKII(I-1,J)+RKJJ(I,J)+RKJJ(I,J-1))
      END DO
      END DO
C
      ITR=0
C
  350 CONTINUE
      ITR=ITR+1
      RSQHI=0.
      RSQHJ=0.
C
      DO N=1,NRED
      I=IRED(N)
      J=JRED(N)
      RSD=RKII(I,J)*HI(I+1,J)+RKII(I-1,J)*HI(I-1,J)
     $   +RKJJ(I,J)*HI(I,J+1)+RKJJ(I,J-1)*HI(I,J-1)
     $   +CIJ(I,J)*HI(I,J)
      HI(I,J)=HI(I,J)-RPH*RSD/CIJ(I,J)
      RSQHI=RSQHI+RSD*RSD
      RSD=RKII(I,J)*HJ(I+1,J)+RKII(I-1,J)*HJ(I-1,J)
     $   +RKJJ(I,J)*HJ(I,J+1)+RKJJ(I,J-1)*HJ(I,J-1)
     $   +CIJ(I,J)*HJ(I,J)
      HJ(I,J)=HJ(I,J)-RPH*RSD/CIJ(I,J)
      RSQHJ=RSQHJ+RSD*RSD
      END DO
C
      DO N=1,NBLK
      I=IBLK(N)
      J=JBLK(N)
      RSD=RKII(I,J)*HI(I+1,J)+RKII(I-1,J)*HI(I-1,J)
     $   +RKJJ(I,J)*HI(I,J+1)+RKJJ(I,J-1)*HI(I,J-1)
     $   +CIJ(I,J)*HI(I,J)
      HI(I,J)=HI(I,J)-RPX*RSD/CIJ(I,J)
      RSQHI=RSQHI+RSD*RSD
      RSD=RKII(I,J)*HJ(I+1,J)+RKII(I-1,J)*HJ(I-1,J)
     $   +RKJJ(I,J)*HJ(I,J+1)+RKJJ(I,J-1)*HJ(I,J-1)
     $   +CIJ(I,J)*HJ(I,J)
      HJ(I,J)=HJ(I,J)-RPX*RSD/CIJ(I,J)
      RSQHJ=RSQHJ+RSD*RSD
      END DO
C
      IF (ITR.GE.ITRHM) GO TO 351
      IF (RSQHI.GE.RSQHM.OR.RSQHJ.GE.RSQHM) GO TO 350
C
  351 CONTINUE
      WRITE(6,603)ITR,RSQHI,RSQHJ
      WRITE(66,603)ITR,RSQHI,RSQHJ
C
      DO J=JMIN,JMAX
      DO I=IMIN,IMAX
      RKI(I,J)=HJ(I,J)/HI(I,J)
      RKJ(I,J)=HI(I,J)/HJ(I,J)
      END DO
      END DO
C
      DO J=JMIN,JMAX
      DO I=IMIN,IMAX-1
      RKII(I,J)=0.5*(RKI(I+1,J)+RKI(I,J))
      END DO
      END DO
C
      DO J=JMIN,JMAX-1
      DO I=IMIN,IMAX
      RKJJ(I,J)=0.5*(RKJ(I,J+1)+RKJ(I,J))
      END DO
      END DO
C
      END IF
C
  603 FORMAT(1X,'DIFF HI & HJ, ITER = ',I3,' RSI,RSJ =',2(2X,E11.4))
C
C--------------------------------------------------------------------C
C
C **  SOLVE FOR X AND Y IN THE INTERIOR 
C
      DO J=JMIN,JMAX
      DO I=IMIN,IMAX
      CIJ(I,J)=-(RKII(I,J)+RKII(I-1,J)+RKJJ(I,J)+RKJJ(I,J-1))
      END DO
      END DO
C
      ITR=0
C
  400 CONTINUE
      ITR=ITR+1
      RSQX=0.
      RSQY=0.
C
      DO N=1,NRED
      I=IRED(N)
      J=JRED(N)
      RSD=RKII(I,J)*X(I+1,J)+RKII(I-1,J)*X(I-1,J)
     $    +RKJJ(I,J)*X(I,J+1)+RKJJ(I,J-1)*X(I,J-1)
     $    +CIJ(I,J)*X(I,J)
      X(I,J)=X(I,J)-RPX*RSD/CIJ(I,J)
      RSQX=RSQX+RSD*RSD
      RSD=RKII(I,J)*Y(I+1,J)+RKII(I-1,J)*Y(I-1,J)
     $    +RKJJ(I,J)*Y(I,J+1)+RKJJ(I,J-1)*Y(I,J-1)
     $    +CIJ(I,J)*Y(I,J)
      Y(I,J)=Y(I,J)-RPX*RSD/CIJ(I,J)
      RSQY=RSQY+RSD*RSD
      END DO
C
      DO N=1,NBLK
      I=IBLK(N)
      J=JBLK(N)
      RSD=RKII(I,J)*X(I+1,J)+RKII(I-1,J)*X(I-1,J)
     $    +RKJJ(I,J)*X(I,J+1)+RKJJ(I,J-1)*X(I,J-1)
     $    +CIJ(I,J)*X(I,J)
      X(I,J)=X(I,J)-RPX*RSD/CIJ(I,J)
      RSQX=RSQX+RSD*RSD
      RSD=RKII(I,J)*Y(I+1,J)+RKII(I-1,J)*Y(I-1,J)
     $    +RKJJ(I,J)*Y(I,J+1)+RKJJ(I,J-1)*Y(I,J-1)
     $    +CIJ(I,J)*Y(I,J)
      Y(I,J)=Y(I,J)-RPX*RSD/CIJ(I,J)
      RSQY=RSQY+RSD*RSD
      END DO
C
      IF (ITR.GE.ITRXM)GO TO 401
      IF(RSQX.GE.RSQXM.OR.RSQY.GE.RSQXM)GO TO 400
C
  401 CONTINUE
      WRITE(6,604)ITR,RSQX,RSQY
      WRITE(66,604)ITR,RSQX,RSQY
C
  604 FORMAT(1X,'DIFF X & Y, ITER = ',I3,' RSX,RSY =',2(2X,E11.4))
C
C--------------------------------------------------------------------C
C
C **  EVALUATE CONVERGENCE BY COMPARING CALCULATED AND INTERPOLATED
C **  RKI OR HI AND HJ AT INTERIOR POINTS OR BY EVALUATING THE 
C **  DEVIATION FORM ORTHOGONALITY AT THE GRID POINTS RELATIVE TO A
C **  SPECIFIED VALUE\CAANGORO
C
      IF (ISIRKI.EQ.1) THEN
C
      AJBM=0.
      AGIJMA=0.
      AGIJMI=1.E+10
      ANGMAX=0.
      ANGMIN=1.E+10
      RSQKI=0.
C
      DO N=1,NRED
      I=IRED(N)
      J=JRED(N)
      DXDI=0.5*(X(I+1,J)-X(I-1,J))
      DYDI=0.5*(Y(I+1,J)-Y(I-1,J))
      DXDJ=0.5*(X(I,J+1)-X(I,J-1))
      DYDJ=0.5*(Y(I,J+1)-Y(I,J-1))
      GII=DXDI*DXDI+DYDI*DYDI
      GJJ=DXDJ*DXDJ+DYDJ*DYDJ
      GIJ=DXDI*DXDJ+DYDI*DYDJ
      AGIJ=ABS(1.-GIJ*GIJ/(GII*GJJ))
      ANGD=(ACOS(GIJ/SQRT(GII*GJJ)))*57.29578
      ANGERR=ABS(90.-ANGD)
      ANGMAX=MAX(ANGMAX,ANGERR)
      ANGMIN=MIN(ANGMIN,ANGERR)
      HICAL=SQRT(GII)
      HJCAL=SQRT(GJJ)
      HIHJCAL=HICAL*HJCAL
      RKICAL=HJCAL/HICAL
      RSQKI=RSQKI+(RKICAL-RKI(I,J))**2
      END DO
C
      DO N=1,NBLK
      I=IBLK(N)
      J=JBLK(N)
      DXDI=0.5*(X(I+1,J)-X(I-1,J))
      DYDI=0.5*(Y(I+1,J)-Y(I-1,J))
      DXDJ=0.5*(X(I,J+1)-X(I,J-1))
      DYDJ=0.5*(Y(I,J+1)-Y(I,J-1))
      GII=DXDI*DXDI+DYDI*DYDI
      GJJ=DXDJ*DXDJ+DYDJ*DYDJ
      GIJ=DXDI*DXDJ+DYDI*DYDJ
      AGIJ=ABS(1.-GIJ*GIJ/(GII*GJJ))
      ANGD=(ACOS(GIJ/SQRT(GII*GJJ)))*57.29578
      ANGERR=ABS(90.-ANGD)
      ANGMAX=MAX(ANGMAX,ANGERR)
      ANGMIN=MIN(ANGMIN,ANGERR)
      HICAL=SQRT(GII)
      HJCAL=SQRT(GJJ)
      HIHJCAL=HICAL*HJCAL
      RKICAL=HJCAL/HICAL
      RSQKI=RSQKI+(RKICAL-RKI(I,J))**2
      END DO
C
      WRITE(6,605)ITRG
      WRITE(6,606)RSQKI
      WRITE(6,609)ANGMIN,ANGMAX
      WRITE(66,605)ITRG
      WRITE(66,606)RSQKI
      WRITE(66,609)ANGMIN,ANGMAX
      IF(ITRG.GE.ITRGM)GO TO 1001
C     IF(RSQKI.GE.RSQKIM)GO TO 1000
      IF(ANGMAX.GE.ANGORO)GO TO 1000
C
      END IF
C
      IF (ISIHIHJ.EQ.1) THEN
C
      RSQHI=0.
      RSQHJ=0.
      AJBM=0.
      AGIJMA=0.
      AGIJMI=1.E+10
      ANGMAX=0.
      ANGMIN=1.E+10
C
      DO N=1,NRED
      I=IRED(N)
      J=JRED(N)
      DXDI=0.5*(X(I+1,J)-X(I-1,J))
      DYDI=0.5*(Y(I+1,J)-Y(I-1,J))
      DXDJ=0.5*(X(I,J+1)-X(I,J-1))
      DYDJ=0.5*(Y(I,J+1)-Y(I,J-1))
      GII=DXDI*DXDI+DYDI*DYDI
      GJJ=DXDJ*DXDJ+DYDJ*DYDJ
      GIJ=DXDI*DXDJ+DYDI*DYDJ
      AGIJ=ABS(1.-GIJ*GIJ/(GII*GJJ))
      ANGD=(ACOS(GIJ/SQRT(GII*GJJ)))*57.29578
      ANGERR=ABS(90.-ANGD)
      ANGMAX=MAX(ANGMAX,ANGERR)
      ANGMIN=MIN(ANGMIN,ANGERR)
      HICAL=SQRT(GII)
      HJCAL=SQRT(GJJ)
      RSQHI=RSQHI+(HICAL-HI(I,J))**2
      RSQHJ=RSQHJ+(HJCAL-HJ(I,J))**2
      END DO
C
      DO N=1,NBLK
      I=IBLK(N)
      J=JBLK(N)
      DXDI=0.5*(X(I+1,J)-X(I-1,J))
      DYDI=0.5*(Y(I+1,J)-Y(I-1,J))
      DXDJ=0.5*(X(I,J+1)-X(I,J-1))
      DYDJ=0.5*(Y(I,J+1)-Y(I,J-1))
      GII=DXDI*DXDI+DYDI*DYDI
      GJJ=DXDJ*DXDJ+DYDJ*DYDJ
      GIJ=DXDI*DXDJ+DYDI*DYDJ
      AGIJ=ABS(1.-GIJ*GIJ/(GII*GJJ))
      ANGD=(ACOS(GIJ/SQRT(GII*GJJ)))*57.29578
      ANGERR=ABS(90.-ANGD)
      ANGMAX=MAX(ANGMAX,ANGERR)
      ANGMIN=MIN(ANGMIN,ANGERR)
      HICAL=SQRT(GII)
      HJCAL=SQRT(GJJ)
      RSQHI=RSQHI+(HICAL-HI(I,J))**2
      RSQHJ=RSQHJ+(HJCAL-HJ(I,J))**2
      END DO
C
      WRITE(6,605)ITRG
      WRITE(6,607)RSQHI,RSQHJ
      WRITE(6,609)ANGMIN,ANGMAX
      WRITE(66,605)ITRG
      WRITE(66,607)RSQHI,RSQHJ
      WRITE(66,609)ANGMIN,ANGMAX
      IF(ITRG.GE.ITRGM)GO TO 1001
C     IF(RSQHI.GE.RSQHIM)GO TO 1000
C     IF(RSQHJ.GE.RSQHJM)GO TO 1000
      IF(ANGMAX.GE.ANGORO)GO TO 1000
C
      END IF 
C
 1001 CONTINUE
      GO TO 3000
C
  605 FORMAT(1X,'GRID GENERATION LOOP ITERATION =',I10)
  606 FORMAT(1X,'GLOBAL RES SQ DIFF IN RKI=',E12.4)
  607 FORMAT(1X,'GLOBAL RES SQ DIFF IN HI,HJ=',2(2X,E12.4))
  609 FORMAT(1X,'MIN AND MAX DEVIATION FROM ORTHO =',2(2X,E12.4))
C
C********************************************************************C
C
C **  NTYPE = 6 GRID GENERATION LOOP 
C
C--------------------------------------------------------------------C
C
 2000 CONTINUE
C
      ITRG=ITRG+1    
C
C--------------------------------------------------------------------C
C
C **  SOLVE FOR X AND Y IN THE INTERIOR 
C
      DO J=JMIN,JMAX
      DO I=IMIN,IMAX
      CIJ(I,J)=-(RKII(I,J)+RKII(I-1,J)+RKJJ(I,J)+RKJJ(I,J-1))
      END DO
      END DO
C
      ITR=0
C
 2400 CONTINUE
      ITR=ITR+1
      RSQX=0.
      RSQY=0.
C
      DO N=1,NRED
      I=IRED(N)
      J=JRED(N)
      RSD=RKII(I,J)*X(I+1,J)+RKII(I-1,J)*X(I-1,J)
     $    +RKJJ(I,J)*X(I,J+1)+RKJJ(I,J-1)*X(I,J-1)
     $    +CIJ(I,J)*X(I,J)
      X(I,J)=X(I,J)-RPX*RSD/CIJ(I,J)
      RSQX=RSQX+RSD*RSD
      RSD=RKII(I,J)*Y(I+1,J)+RKII(I-1,J)*Y(I-1,J)
     $    +RKJJ(I,J)*Y(I,J+1)+RKJJ(I,J-1)*Y(I,J-1)
     $    +CIJ(I,J)*Y(I,J)
      Y(I,J)=Y(I,J)-RPX*RSD/CIJ(I,J)
      RSQY=RSQY+RSD*RSD
      END DO
C
      DO N=1,NBLK
      I=IBLK(N)
      J=JBLK(N)
      RSD=RKII(I,J)*X(I+1,J)+RKII(I-1,J)*X(I-1,J)
     $    +RKJJ(I,J)*X(I,J+1)+RKJJ(I,J-1)*X(I,J-1)
     $    +CIJ(I,J)*X(I,J)
      X(I,J)=X(I,J)-RPX*RSD/CIJ(I,J)
      RSQX=RSQX+RSD*RSD
      RSD=RKII(I,J)*Y(I+1,J)+RKII(I-1,J)*Y(I-1,J)
     $    +RKJJ(I,J)*Y(I,J+1)+RKJJ(I,J-1)*Y(I,J-1)
     $    +CIJ(I,J)*Y(I,J)
      Y(I,J)=Y(I,J)-RPX*RSD/CIJ(I,J)
      RSQY=RSQY+RSD*RSD
      END DO
C
      IF (ITR.GE.ITRXM)GO TO 2401
      IF(RSQX.GE.RSQXM.OR.RSQY.GE.RSQXM)GO TO 2400
C
 2401 CONTINUE
      WRITE(6,604)ITR,RSQX,RSQY
      WRITE(66,604)ITR,RSQX,RSQY
C
C--------------------------------------------------------------------C
C
C **  EVALUATE CONVERGENCE BY EVALUATING THE 
C **  DEVIATION FORM ORTHOGONALITY AT THE GRID POINTS RELATIVE TO A
C **  SPECIFIED VALUE\CAANGORO
C
      AGIJMA=0.
      AGIJMI=1.E+10
      ANGMAX=0.
      ANGMIN=1.E+10
C
      DO J=JMINO,JMAXO-1
      DO I=IMINO,IMAXO-1
      IF (IJCT(I,J).GE.1.AND.IJCT(I,J).LE.5) THEN
        DXDI=0.5*(X(I+1,J+1)-X(I,J+1)+X(I+1,J)-X(I,J))
        DYDI=0.5*(Y(I+1,J+1)-Y(I,J+1)+Y(I+1,J)-Y(I,J))
        DXDJ=0.5*(X(I+1,J+1)-X(I+1,J)+X(I,J+1)-X(I,J))
        DYDJ=0.5*(Y(I+1,J+1)-Y(I+1,J)+Y(I,J+1)-Y(I,J))
        GII=DXDI*DXDI+DYDI*DYDI
        GJJ=DXDJ*DXDJ+DYDJ*DYDJ
        GIJ=DXDI*DXDJ+DYDI*DYDJ
        RKI(I,J)=GJJ
        RKJ(I,J)=GII
        ANGD=(ACOS(GIJ/SQRT(GII*GJJ)))*57.29578
        ANGERR=ABS(90.-ANGD)
        ANGMAX=MAX(ANGMAX,ANGERR)
        ANGMIN=MIN(ANGMIN,ANGERR)
      END IF
      END DO
      END DO
C
      DO J=JMIN+1,JMAX-1
      DO I=IMIN,IMAX-1
      RKII(I,J)=0.5*(RKI(I,J)+RKI(I,J-1))
      END DO
      END DO
C
      DO J=JMIN,JMAX-1
      DO I=IMIN+1,IMAX
      RKJJ(I,J)=0.5*(RKJ(I,J)+RKJ(I-1,J))
      END DO
      END DO
C
      WRITE(6,605)ITRG
      WRITE(6,609)ANGMIN,ANGMAX
      WRITE(66,605)ITRG
      WRITE(66,609)ANGMIN,ANGMAX
      IF(ITRG.GE.ITRGM)GO TO 2001
      IF(ANGMAX.GE.ANGORO)GO TO 2000
C
 2001 CONTINUE
      GO TO 3000
C
C********************************************************************C
C
C **  NTYPE = 7 GRID GENERATION LOOP 
C
 7000 CONTINUE
C
      ITN7=0
C
C--------------------------------------------------------------------C
C
C **  SET CORNER COORDINATES
C
      X(IB,JB)=XIBJB
      Y(IB,JB)=YIBJB
      X(IE,JB)=XIEJB
      Y(IE,JB)=YIEJB
      X(IB,JE)=XIBJE
      Y(IB,JE)=YIBJE
      X(IE,JE)=XIEJE
      Y(IE,JE)=YIEJE
C
      XN(IB,JB)=XIBJB
      YN(IB,JB)=YIBJB
      XN(IE,JB)=XIEJB
      YN(IE,JB)=YIEJB
      XN(IB,JE)=XIBJE
      YN(IB,JE)=YIBJE
      XN(IE,JE)=XIEJE
      YN(IE,JE)=YIEJE
C
C--------------------------------------------------------------------C
C
C **  INITIALIZE X AND Y ALONG I=IB AND I=IE
C
      DYTMP=(YIBJE-YIBJB)/FLOAT(JE-JB)
      YTMP=YIBJB
      DO J=JB+1,JE-1
      YTMP=YTMP+DYTMP
      Y(IB,J)=YTMP
      X(IB,J)=FIB(YTMP,J)
      END DO
C
      DYTMP=(YIEJE-YIEJB)/FLOAT(JE-JB)
      YTMP=YIEJB
      DO J=JB+1,JE-1
      YTMP=YTMP+DYTMP
      Y(IE,J)=YTMP
      X(IE,J)=FIE(YTMP,J)
      END DO
C
C--------------------------------------------------------------------C
C
C **  INITIALIZE X AND Y ALONG J=JB AND J=JE
C
      DXTMP=(XIEJB-XIBJB)/FLOAT(IE-IB)
      XTMP=XIBJB
      DO I=IB+1,IE-1
      XTMP=XTMP+DXTMP
      X(I,JB)=XTMP
      Y(I,JB)=GJB(XTMP,I)
      END DO
C
      DXTMP=(XIEJE-XIBJE)/FLOAT(IE-IB)
      XTMP=XIBJE
      DO I=IB+1,IE-1
      XTMP=XTMP+DXTMP
      X(I,JE)=XTMP
      Y(I,JE)=GJE(XTMP,I)
      END DO
C
C--------------------------------------------------------------------C
C
      OPEN(90,FILE='ibndry.out',STATUS='UNKNOWN')
C
      WRITE(90,9301)
      J=JB
      DO I=IB,IE
      WRITE(90,9300)I,J,X(I,J),Y(I,J)
      END DO
      WRITE(90,9305)
      WRITE(90,9302)
      I=IE
      DO J=JB,JE
      WRITE(90,9300)I,J,X(I,J),Y(I,J)
      END DO
      WRITE(90,9305)
      WRITE(90,9303)
      J=JE
      DO I=IE,IB,-1
      WRITE(90,9300)I,J,X(I,J),Y(I,J)
      END DO
      WRITE(90,9305)
      WRITE(90,9304)
      I=IB
      DO J=JE,JB,-1
      WRITE(90,9300)I,J,X(I,J),Y(I,J)
      END DO
C
      CLOSE(90)
C
 9300 FORMAT(2I5,2F12.4)
 9301 FORMAT(' ALONG J=JB ',/)
 9302 FORMAT(' ALONG I=IE ',/)
 9303 FORMAT(' ALONG J=JE ',/)
 9304 FORMAT(' ALONG I=IB ',/)
 9305 FORMAT(/)
 9306 FORMAT(2F12.3)
C
C--------------------------------------------------------------------C
C
C **  FILL THE INTERIOR BY RELAXATION
C
      SMD=1.
      SSQ=SMD*SMD
      CR7=.5/(1.+SSQ)
C
C **  RELAXATION LOOP
C
      NR=0
 7110 CONTINUE
      NR=NR+1
      DO J=JB+1,JE-1
      DO I=IB+1,IE-1
      XN(I,J)=CR7*( X(I+1,J)+X(I-1,J)+SSQ*( X(I,J+1)+X(I,J-1) ) )
      YN(I,J)=CR7*( Y(I+1,J)+Y(I-1,J)+SSQ*( Y(I,J+1)+Y(I,J-1) ) )
      END DO
      END DO
      DO J=JB+1,JE-1
      DO I=IB+1,IE-1
      X(I,J)=XN(I,J)
      Y(I,J)=YN(I,J)
      END DO
      END DO
      IF(NR.LT.N7RELAX) GO TO 7110
      WRITE(6,7666)NR   
C
C **  CALCULATE CONFORMAL MODULE SMD
C
      CALL CALSMD
      SMDOLD=SMD
      SSQ=SMD*SMD
      SMDI=1./SMD
      CR7=.5/(1.+SSQ)
      WRITE(6,7667)SMDOLD,SMD
C
 7666 FORMAT(' COMPLETED INIT, NR  = ',I10/)
 7667 FORMAT(' SMDOLD,SMD AFTER INIT     = ',2E14.5/)
 7668 FORMAT(' SMDOLD,SMD AFTER X SWEEP  = ',2E14.5/)
 7669 FORMAT(' SMDOLD,SMD AFTER Y SWEEP  = ',2E14.5/)
C
C--------------------------------------------------------------------C
C
C **  WRITE DXY FILE OF INITIAL GRID
C
C **  BEGIN DXF WRITE SEQUENCE TO FILE init.dxf
C       The following section added by M.R. Morton, Tetra Tech:
C       Write coordinates of each cell corner to DXF format file.
C       Each cell will be a closed polyline in the DXF file:
C       IJCT = 1, triangular computational cell in NE quadrant:
      DO J=1,JE
      DO I=1,IE 
        if (IJCT(I,J) .eq. 1) then                     
          write(26,1650)                               
          write(26,1651) X(I  ,J  ),Y(I  ,J  )         
          write(26,1651) X(I+1,J  ),Y(I+1,J  )         
          write(26,1651) X(I  ,J+1),Y(I  ,J+1)         
          write(26,1651) X(I  ,J  ),Y(I  ,J  )         
          write(26,1652)                               
        end if                                         
C       IJCT = 4, computational cell in SE quadrant:       
        if (IJCT(I,J) .eq. 4) then                     
          write(26,1650)                               
          write(26,1651) X(I  ,J  ),Y(I  ,J  )         
          write(26,1651) X(I+1,J  ),Y(I+1,J  )         
          write(26,1651) X(I+1,J+1),Y(I+1,J+1)         
          write(26,1651) X(I  ,J  ),Y(I  ,J  )         
          write(26,1652)                               
        end if                                         
C       IJCT = 3, computational cell in SW quadrant:       
        if (IJCT(I,J) .eq. 3) then                     
          write(26,1650)                               
          write(26,1651) X(I+1,J  ),Y(I+1,J  )         
          write(26,1651) X(I+1,J+1),Y(I+1,J+1)         
          write(26,1651) X(I  ,J+1),Y(I  ,J+1)         
          write(26,1651) X(I+1,J  ),Y(I+1,J  )         
          write(26,1652)                               
        end if                                         
C       IJCT = 2, computational cell in NW quadrant:       
        if (IJCT(I,J) .eq. 2) then                     
          write(26,1650)                               
          write(26,1651) X(I  ,J  ),Y(I  ,J  )         
          write(26,1651) X(I+1,J+1),Y(I+1,J+1)         
          write(26,1651) X(I  ,J+1),Y(I  ,J+1)         
          write(26,1651) X(I  ,J  ),Y(I  ,J  )         
          write(26,1652)                               
        end if                                         
C       IJCT = 5, normal four-sided water cell:            
        if (IJCT(I,J) .eq. 5) then                     
          write(26,1650)                               
          write(26,1651) X(I  ,J  ),Y(I  ,J  )         
          write(26,1651) X(I+1,J  ),Y(I+1,J  )         
          write(26,1651) X(I+1,J+1),Y(I+1,J+1)         
          write(26,1651) X(I  ,J+1),Y(I  ,J+1)         
          write(26,1651) X(I  ,J  ),Y(I  ,J  )         
          write(26,1652)                               
        end if
      END DO
      END DO
C **  END DXF WRITE SEQUENCE TO FILE grid.dxf   
C
C--------------------------------------------------------------------C
C
C **  BEGIN GENERATION PROCEDURE
C
 7120 CONTINUE
      ITN7=ITN7+1
C
C **  SET X EQUATION BOUNDARY CONDITIONS
C
      DO J=JB+1,JE-1
       X(IB,J)=FIB( Y(IB,J), J )
       X(IE,J)=FIE( Y(IE,J), J )
      END DO
C
C **  ITERATE INTERIOR X EQUATIONS
C
      DO NXY=1,NXYIT
C
      DO I=IB+1,IE-1
      IF(MOD(I+JB,2).EQ.0) THEN
        RESD7=X(I,JB)-X(I,JB+1)
     $       -0.25*SMDI*( Y(I+1,JB  )-Y(I-1,JB  ) 
     $                   +Y(I+1,JB+1)-Y(I-1,JB+1) )
        X(I,JB)=X(I,JB)-RP7*RESD7
      END IF
      IF(MOD(I+JE,2).EQ.0) THEN
        RESD7=X(I,JE)-X(I,JE-1)
     $       +0.25*SMDI*( Y(I+1,JE  )-Y(I-1,JE  )
     $                   +Y(I+1,JE-1)-Y(I-1,JE-1) )
        X(I,JE)=X(I,JE)-RP7*RESD7
      END IF
      END DO
C
      DO J=JB+1,JE-1
       DO I=IB+1,IE-1
       IF(MOD(I+J,2).EQ.0) THEN
         RESD7= X(I,J)-CR7*( X(I+1,J)+X(I-1,J)
     $        +SSQ*(X(I,J+1)+X(I,J-1)) )
         X(I,J)=X(I,J)-RP7*RESD7 
       END IF
       END DO
      END DO
C
      DO I=IB+1,IE-1
      IF(MOD(I+JB,2).NE.0) THEN
        RESD7=X(I,JB)-X(I,JB+1)
     $       -0.25*SMDI*( Y(I+1,JB  )-Y(I-1,  JB)
     $                   +Y(I+1,JB+1)-Y(I-1,JB+1) )
        X(I,JB)=X(I,JB)-RP7*RESD7
      END IF
      IF(MOD(I+JE,2).NE.0) THEN
        RESD7=X(I,JE)-X(I,JE-1)
     $       +0.25*SMDI*( Y(I+1,JE  )-Y(I-1,JE  )
     $                   +Y(I+1,JE-1)-Y(I-1,JE-1) )
        X(I,JE)=X(I,JE)-RP7*RESD7
      END IF
      END DO
C
      DO J=JB+1,JE-1
       DO I=IB+1,IE-1
       IF(MOD(I+J,2).NE.0) THEN
         RESD7=X(I,J)-CR7*( X(I+1,J)+X(I-1,J)
     $        +SSQ*(X(I,J+1)+X(I,J-1)) )
         X(I,J)=X(I,J)-RP7*RESD7 
       END IF   
       END DO
      END DO
C
C     DO J=JB+1,JE-1
C      DO I=IB+1,IE-1
C       XN(I,J)=CR7*( X(I+1,J)+X(I-1,J)+SSQ*(X(I,J+1)+X(I,J-1)) ) 
C      END DO
C     END DO
C  
C     DO I=IB+1,IE-1
C      XN(I,JB)=( 4.*X(I,JB+1)-X(I,JB+2)
C    $       +SMDI*(Y(I+1,JB)-Y(I-1,JB)) )/3.
C      XN(I,JE)=( 4.*X(I,JE-1)-X(I,JE-2)
C    $       -SMDI*(Y(I+1,JE)-Y(I-1,JE)) )/3.
C     END DO
C
C     DO J=JB+1,JE-1
C      DO I=IB+1,IE-1
C       X(I,J)=XN(I,J)
C      END DO
C     END DO
C  
C     DO I=IB+1,IE-1
C      X(I,JB)=XN(I,JB)
C      X(I,JE)=XN(I,JE)
C     END DO
C
      END DO
C
C **  END INTERIOR X EQUATION ITERATION
C
C **  UPDATE CONFORMAL MODULE SMD
C
      SMDOLD=SMD
      CALL CALSMD
      SSQ=SMD*SMD
      SMDI=1./SMD
      CR7=.5/(1.+SSQ)
      WRITE(6,7668)SMDOLD,SMD
C
C **  SET Y EQUATION BOUNDARY CONDITIONS
C
      DO I=IB+1,IE-1
       Y(I,JB)=GJB( X(I,JB), I )
       Y(I,JE)=GJE( X(I,JE), I )
      END DO
C
C **  ITERATE INTERIOR Y EQUATIONS
C
      DO NXY=1,NXYIT
C
      DO J=JB+1,JE-1
      IF(MOD(IB+J,2).EQ.0) THEN
        RESD7=Y(IB,J)-Y(IB+1,J)
     $        -0.25*SMD*( X(IB  ,J+1)-X(IB  ,J-1)
     $                   +X(IB+1,J+1)-X(IB+1,J-1) )
        Y(IB,J)=Y(IB,J)-RP7*RESD7
      END IF
      IF(MOD(IE+J,2).EQ.0) THEN
        RESD7=Y(IE,J)-Y(IE-1,J)
     $        +0.25*SMD*( X(IE  ,J+1)-X(IE  ,J-1)
     $                   +X(IE-1,J+1)-X(IE-1,J-1) )
        Y(IE,J)=Y(IE,J)-RP7*RESD7
      END IF
      END DO
C
      DO J=JB+1,JE-1
       DO I=IB+1,IE-1
       IF(MOD(I+J,2).EQ.0) THEN
         RESD7=Y(I,J)-CR7*( Y(I+1,J)+Y(I-1,J)
     $        +SSQ*(Y(I,J+1)+Y(I,J-1)) )
         Y(I,J)=Y(I,J)-RP7*RESD7 
       END IF   
       END DO
      END DO
C
      DO J=JB+1,JE-1
      IF(MOD(IB+J,2).NE.0) THEN
        RESD7=Y(IB,J)-Y(IB+1,J)
     $        -0.25*SMD*( X(IB  ,J+1)-X(IB  ,J-1)
     $                   +X(IB+1,J+1)-X(IB+1,J-1) )
        Y(IB,J)=Y(IB,J)-RP7*RESD7
      END IF
      IF(MOD(IE+J,2).NE.0) THEN
        RESD7=Y(IE,J)-Y(IE-1,J)
     $        +0.25*SMD*( X(IE  ,J+1)-X(IE  ,J-1)
     $                   +X(IE-1,J+1)-X(IE-1,J-1) )
        Y(IE,J)=Y(IE,J)-RP7*RESD7
      END IF
      END DO
C
      DO J=JB+1,JE-1
       DO I=IB+1,IE-1
       IF(MOD(I+J,2).NE.0) THEN
         RESD7= Y(I,J)-CR7*( Y(I+1,J)+Y(I-1,J)
     $        +SSQ*(Y(I,J+1)+Y(I,J-1)) )
         Y(I,J)=Y(I,J)-RP7*RESD7 
       END IF   
       END DO
      END DO
C
C     DO J=JB+1,JE-1
C      DO I=IB+1,IE-1
C       YN(I,J)=CR7*( Y(I+1,J)+Y(I-1,J)+SSQ*(Y(I,J+1)+Y(I,J-1)) )
C      END DO
C     END DO
C
C     DO J=JB+1,JE-1
C      YN(IB,J)=( 4.*Y(IB+1,J)-Y(IB+2,J)
C    $        +SMD*(X(IB,J+1)-X(IB,J-1)) )/3.
C      YN(IE,J)=( 4.*Y(IE-1,J)-Y(IE-2,J)
C    $        -SMD*(X(IE,J+1)-X(IE,J-1)) )/3.
C     END DO
C
C     DO J=JB+1,JE-1
C      DO I=IB+1,IE-1
C       Y(I,J)=YN(I,J)
C      END DO
C     END DO
C
C     DO J=JB+1,JE-1
C      Y(IB,J)=YN(IB,J)
C      Y(IE,J)=YN(IE,J)
C     END DO
C
      END DO
C
C **  END INTERIOR Y EQUATION ITERATION
C
C **  UPDATE CONFORMAL MODULE SMD
C
      SMDOLD=SMD
      CALL CALSMD
      SSQ=SMD*SMD
      SMDI=1./SMD
      CR7=.5/(1.+SSQ)
      WRITE(6,7669)SMDOLD,SMD
C
C **  CHECK CONVERGENCE BASED ON SMD
C
      SERR=ABS( (SMDOLD-SMD)/SMD )
C
C **  EVALUATE CONVERGENCE BY EVALUATING THE 
C **  DEVIATION FORM ORTHOGONALITY AT THE GRID POINTS RELATIVE TO A
C **  SPECIFIED VALUE\CAANGORO
C
      AGIJMA=0.
      AGIJMI=1.E+10
      ANGMAX=0.
      ANGMIN=1.E+10
C
      DO J=JB,JE-1
      DO I=IB,IE-1
        DXDI=0.5*(X(I+1,J+1)-X(I,J+1)+X(I+1,J)-X(I,J))
        DYDI=0.5*(Y(I+1,J+1)-Y(I,J+1)+Y(I+1,J)-Y(I,J))
        DXDJ=0.5*(X(I+1,J+1)-X(I+1,J)+X(I,J+1)-X(I,J))
        DYDJ=0.5*(Y(I+1,J+1)-Y(I+1,J)+Y(I,J+1)-Y(I,J))
        GII=DXDI*DXDI+DYDI*DYDI
        GJJ=DXDJ*DXDJ+DYDJ*DYDJ
        GIJ=DXDI*DXDJ+DYDI*DYDJ
        ANGD=(ACOS(GIJ/SQRT(GII*GJJ)))*57.29578
        ANGERR=ABS(90.-ANGD)
        ANGMAX=MAX(ANGMAX,ANGERR)
        ANGMIN=MIN(ANGMIN,ANGERR)
       END DO
      END DO
C
      WRITE(6,7881)ITN7,ITN7MAX
      WRITE(6,7882)SMDOLD,SMD,SERR,SERRMAX
      WRITE(6,7883)ANGMIN,ANGMAX
      IF(ITN7.GT.ITN7MAX) GO TO 2999
      IF(ANGMAX.LE.ANGORO)GO TO 2999
      IF(SERR.GT.SERRMAX) GO TO 7120
C
 2999 CONTINUE
C
C **  WRITE SHOREMASK FILE
C
C*****BEGIN HARDWIRED ADDITION FOR WCA GRID ONLY*********************C
      X(21,3)=13.35
      Y(21,3)=8.68
      X(22,3)=15.14
      Y(22,3)=6.7
      X(23,3)=17.9
      Y(23,3)=5.8
      X(24,3)=20.08
      Y(24,3)=5.97
      X(22,2)=15.1
      Y(22,2)=3.8
      X(23,2)=16.0
      Y(23,2)=3.5
C*****END HARDWIRED ADDITION FOR WCA GRID ONLY***********************C
C
      OPEN(90,FILE='shorebndry',STATUS='UNKNOWN')
      OPEN(91,FILE='shoremask',STATUS='UNKNOWN')
      I=IB
      DO J=JB,JE
      WRITE(90,9300)I,J,X(I,J),Y(I,J)
      WRITE(91,9306)X(I,J),Y(I,J)
      END DO
      J=JE
      DO I=IB+1,IE
      WRITE(90,9300)I,J,X(I,J),Y(I,J)
      WRITE(91,9306)X(I,J),Y(I,J)
      END DO
      I=IE
      DO J=JE-1,JB,-1
      WRITE(90,9300)I,J,X(I,J),Y(I,J)
      WRITE(91,9306)X(I,J),Y(I,J)
      END DO
      J=JB
C     DO I=IE-1,IB+1,-1
C     WRITE(90,9300)I,J,X(I,J),Y(I,J)
C     WRITE(91,9306)X(I,J),Y(I,J)
C     END DO
C*****WHEN ELIMINATING THE BELOW HARDWIRE, UNCOMMENT ABOVE 4 LINES***C
C*****BEGIN HARDWIRED ADDITION FOR WCA GRID ONLY*********************C
      DO I=IE-1,24,-1
      WRITE(90,9300)I,J,X(I,J),Y(I,J)
      WRITE(91,9306)X(I,J),Y(I,J)
      END DO
      I=23
      J=3
      WRITE(90,9300)I,J,X(I,J),Y(I,J)
      WRITE(91,9306)X(I,J),Y(I,J)
      I=23
      J=2
      WRITE(90,9300)I,J,X(I,J),Y(I,J)
      WRITE(91,9306)X(I,J),Y(I,J)
      I=22
      J=3
      WRITE(90,9300)I,J,X(I,J),Y(I,J)
      WRITE(91,9306)X(I,J),Y(I,J)
      J=4
      DO I=21,IB,-1
      WRITE(90,9300)I,J,X(I,J),Y(I,J)
      WRITE(91,9306)X(I,J),Y(I,J)
      END DO
C*****END HARDWIRED ADDITION FOR WCA GRID ONLY***********************C
C
      CLOSE(90)
      CLOSE(91)
C
C **  WRITE GRID CORD FILE FOR TYPE 7
C
      DO J=JB,JE
      DO I=IB,IE
      WRITE(12,9306)X(I,J),Y(I,J)
      END DO
      WRITE(12,7884)
      END DO
C
      DO I=IB,IE
      DO J=JB,JE
      WRITE(12,9306)X(I,J),Y(I,J)
      END DO
      WRITE(12,7884)
      END DO
C
C*****BEGIN HARDWIRED ADDITION FOR WCA GRID ONLY*********************C
C
      J=3
      DO I=21,24
      WRITE(12,9306)X(I,J),Y(I,J)
      END DO
      WRITE(12,7884)
C
      J=2
      DO I=22,23
      WRITE(12,9306)X(I,J),Y(I,J)
      END DO
      WRITE(12,7884)
C
      I=21
      DO J=3,4
      WRITE(12,9306)X(I,J),Y(I,J)
      END DO
      WRITE(12,7884)
C
      I=22
      DO J=2,4
      WRITE(12,9306)X(I,J),Y(I,J)
      END DO
      WRITE(12,7884)
C
      I=23
      DO J=2,4
      WRITE(12,9306)X(I,J),Y(I,J)
      END DO
      WRITE(12,7884)
C
      I=24
      DO J=3,4
      WRITE(12,9306)X(I,J),Y(I,J)
      END DO
      WRITE(12,7884)
C
C*****END HARDWIRED ADDITION FOR WCA GRID ONLY***********************C
C
 7881 FORMAT(' NTYPE 7, ITER,ITERMAX = ',2I10/)
 7882 FORMAT(' SMDOLD,SMD,SERR,SERRMAX = ',4E14.5/)
 7883 FORMAT(' ANGMIN,ANGMAX = ',2E14.4/)
 7884 FORMAT(' A ')
C
C********************************************************************C
C
 3000 CONTINUE
C
C********************************************************************C
C
C **  SHIFT COORDINATES
C
      DO J=JMIN,JMAX
      DO I=IMIN,IMAX
      X(I,J)=X(I,J)-XSHIFT
      Y(I,J)=Y(I,J)-YSHIFT
      END DO
      END DO
C
      IF(NTYPE.EQ.7) GO TO 3500
C
C********************************************************************C
C
C **  PROCESS AND OUTPUT RESULTS FOR PLOTTING THE GRID
C
      WRITE(7,79)
      WRITE(7,76)
C
      IF(NTYPE.EQ.1)THEN
       DO J=JMIN,JMAX
       WRITE(9,92)J
       WRITE(12,92)J
       DO I=IMIN,IMAXO
       IF(KSGI(I,J).EQ.1)THEN
        WRITE(7,77)I,J,X(I,J),Y(I,J)
        WRITE(9,90)X(I,J),Y(I,J)
        WRITE(12,90)X(I,J),Y(I,J)
       END IF
       END DO
       END DO
       DO I=IMIN,IMAXO
       WRITE(9,91)I
       WRITE(12,91)I
       DO J=JMIN,JMAX
       IF(KSGI(I,J).EQ.1)THEN
        WRITE(8,90)X(I,J),Y(I,J)
        WRITE(12,90)X(I,J),Y(I,J)
       END IF
       END DO
       END DO
      END IF
C
      IF(NTYPE.EQ.2)THEN
       DO J=JMIN,JMAXO
       WRITE(9,92)J
       WRITE(12,92)J
       DO I=IMIN,IMAX
       IF(KSGI(I,J).EQ.1)THEN
        WRITE(7,77)I,J,X(I,J),Y(I,J)
        WRITE(9,90)X(I,J),Y(I,J)
        WRITE(12,90)X(I,J),Y(I,J)
       END IF
       END DO
       END DO
       DO I=IMIN,IMAX
       WRITE(9,91)I
       WRITE(12,91)I
       DO J=JMIN,JMAXO
       IF(KSGI(I,J).EQ.1)THEN
        WRITE(8,90)X(I,J),Y(I,J)
        WRITE(12,90)X(I,J),Y(I,J)
       END IF
       END DO
       END DO
      END IF
C           
      IF(NTYPE.EQ.3)THEN
       DO J=JMIN,JMAX
       WRITE(9,92)J
       WRITE(12,92)J
       DO I=IMINO,IMAX
       IF(KSGI(I,J).EQ.1)THEN
        WRITE(7,77)I,J,X(I,J),Y(I,J)
        WRITE(9,90)X(I,J),Y(I,J)
        WRITE(12,90)X(I,J),Y(I,J)
       END IF
       END DO
       END DO
       DO I=IMINO,IMAX
       WRITE(9,91)I
       WRITE(12,91)I
       DO J=JMIN,JMAX
       IF(KSGI(I,J).EQ.1)THEN
        WRITE(8,90)X(I,J),Y(I,J)
        WRITE(12,90)X(I,J),Y(I,J)
       END IF
       END DO
       END DO
      END IF
C           
      IF(NTYPE.EQ.4)THEN
       DO J=JMINO,JMAX
       WRITE(9,92)J
       WRITE(12,92)J
       DO I=IMIN,IMAX
       IF(KSGI(I,J).EQ.1)THEN
        WRITE(7,77)I,J,X(I,J),Y(I,J)
        WRITE(8,90)X(I,J),Y(I,J)
        WRITE(12,90)X(I,J),Y(I,J)
       END IF
       END DO
       END DO
       DO I=IMIN,IMAX
       WRITE(9,91)I
       WRITE(12,91)I
       DO J=JMINO,JMAX
       IF(KSGI(I,J).EQ.1)THEN
        WRITE(8,90)X(I,J),Y(I,J)
        WRITE(12,90)X(I,J),Y(I,J)
       END IF
       END DO
       END DO
      END IF
C
      IF(NTYPE.GE.5.AND.NTYPE.LE.6)THEN
       DO J=JMIN,JMAX
       WRITE(9,92)J
       WRITE(12,92)J
       DO I=IMIN,IMAX
       IF(KSGI(I,J).EQ.1)THEN
        WRITE(7,77)I,J,X(I,J),Y(I,J)
        WRITE(9,90)X(I,J),Y(I,J)
        WRITE(12,90)X(I,J),Y(I,J)
       END IF
       END DO
       END DO
       DO I=IMIN,IMAX
       WRITE(9,91)I
       WRITE(12,91)I
       DO J=JMIN,JMAX
       IF(KSGI(I,J).EQ.1)THEN
        WRITE(8,90)X(I,J),Y(I,J)
        WRITE(12,90)X(I,J),Y(I,J)
       END IF
       END DO
       END DO
      END IF
C
   90 FORMAT(1X,F12.4,5X,F12.4)
   91 FORMAT(1X,'I=',I5)
   92 FORMAT(1X,'J=',I5)
C
C********************************************************************C
C
C **  CALCULATE AND OUTPUT THE METRIC COEFFICIENTS FOR THE WATER 
C **  CELLS AND INTERPOLATE DEPTH, BOT ELEV AND VEG TYPE
C **  TO THE CELL CENTERS
C
 3500 CONTINUE
C
C********************************************************************C
C
C **  DEPTH AND VEGATATION DATA PROCESSING
C
C--------------------------------------------------------------------C
C
C **  INITIAL DEPTH AND VEGETATION TYPE INTERPOLATION
C
      IF(ISIDEP.EQ.1.OR.ISVEG.EQ.1) THEN
      DO J=JMINO,JMAXO-1
      DO I=IMINO,IMAXO-1
      IF (IJCT(I,J).GE.1.AND.IJCT(I,J).LE.5) THEN
        DXDI=0.5*(X(I+1,J+1)-X(I,J+1)+X(I+1,J)-X(I,J))
        DYDI=0.5*(Y(I+1,J+1)-Y(I,J+1)+Y(I+1,J)-Y(I,J))
        DXDJ=0.5*(X(I+1,J+1)-X(I+1,J)+X(I,J+1)-X(I,J))
        DYDJ=0.5*(Y(I+1,J+1)-Y(I+1,J)+Y(I,J+1)-Y(I,J))
        GII=DXDI*DXDI+DYDI*DYDI
        GJJ=DXDJ*DXDJ+DYDJ*DYDJ
        HII=SQRT(GII)
        HJJ=SQRT(GJJ)
        RKI(I,J)=HJJ/HII
        RKJ(I,J)=HII/HJJ
        XLNUTME=0.25*(X(I,J)+X(I+1,J)+X(I+1,J+1)+X(I,J+1))
        YLTUTMN=0.25*(Y(I,J)+Y(I+1,J)+Y(I+1,J+1)+Y(I,J+1))
C       RADSQ=HII*HJJ
        HIISQ=HII*HII
        HJJSQ=HJJ*HJJ
        RADSQ1=MIN(HIISQ,HJJSQ)
        RADSQ2=MAX(HIISQ,HJJSQ)
        IF(ISIDEP.EQ.1) THEN
          CALL RADDEP(I,J,XLNUTME,YLTUTMN,RADSQ1,RADSQ2,DEPTH)
          IF(DEPTH.LE.-998.) THEN
            N999=N999+1
            DEPCC(I,J)=0.
            DEPFIX(I,J)=0.
            WRITE(67,6777)I,J,XLNUTME,YLTUTMN
           ELSE
            DEPCC(I,J)=DEPTH
            DEPFIX(I,J)=DEPTH
			IF(DEPTH.GT.1000.) THEN
			  WRITE(67,6779)I,J,DEPTH
			END IF
          END IF
        END IF
        IF(ISVEG.EQ.1) THEN
          CALL RADVEG(I,J,XLNUTME,YLTUTMN,RADSQ1,RADSQ2,NVTMP)
          IF(NVTMP.EQ.0) THEN
            N999=N999+1
            NVEGIJ(I,J)=0
            WRITE(67,6888)I,J,XLNUTME,YLTUTMN
           ELSE
            NVEGIJ(I,J)=NVTMP
          END IF
        END IF
      END IF
      END DO
      END DO
      END IF
C
C--------------------------------------------------------------------C
C
C **  FILL MISSING DEPTH POINTS
C
      IF(ISIDEP.EQ.1) THEN
c
      DO J=JMINO,JMAXO
      DO I=IMINO,IMAXO
      IF (IJCT(I,J).GE.1.AND.IJCT(I,J).LE.5) THEN
        RKII(I,J)=0.5*(RKI(I-1,J)+RKI(I,J))
        RKJJ(I,J)=0.5*(RKJ(I,J-1)+RKJ(I,J))
        IF (IJCT(I-1,J).EQ.9) RKII(I,J)=0.
        IF (IJCT(I,J-1).EQ.9) RKJJ(I,J)=0.
       ELSE
        RKII(I,J)=0.
        RKJJ(I,J)=0.
      END IF
      END DO
      END DO
C
c     DO J=10,63
c     DEPCC(6,J)= 2.5
c     DEPCC(7,J)= 2.5
c     DEPCC(8,J)= 2.5
c     DEPCC(9,J)= 2.5
c     END DO
C
      DO J=JMINO,JMAXO
      DO I=IMINO,IMAXO
       XN(I,J)=DEPCC(I,J)
      END DO
      END DO
C
      DO ND=1,NDEPSM 
       DO J=JMINO,JMAXO
       DO I=IMINO,IMAXO
        IF(IJCT(I,J).GE.1.AND.IJCT(I,J).LE.5) THEN
c       CDTMP=1./( RKII(I+1,J)+RKII(I,J)+RKJJ(I,J+1)+RKJJ(I,J) )
c       DEPCC(I,J)=CDTMP*(RKII(I+1,J)*XN(I+1,J)+RKII(I,J)*XN(I-1,J)
c    $                   +RKJJ(I,J+1)*XN(I,J+1)+RKJJ(I,J)*XN(I,J-1))
        CTOT=RKII(I+1,J)+RKII(I,J)+RKJJ(I,J+1)+RKJJ(I,J)
		CCEN=(1.-0.0625*CTOT)*XN(I,J)
        CAVG=0.0625*(RKII(I+1,J)*XN(I+1,J)+RKII(I,J)*XN(I-1,J)
     $                   +RKJJ(I,J+1)*XN(I,J+1)+RKJJ(I,J)*XN(I,J-1))
        DEPCC(I,J)=CCEN+CAVG
		IF(ND.EQ.NDEPSM) THEN
		  IF(CCEN.LT.0.) WRITE(67,6769)I,J,CTOT
		END IF
       END IF
       END DO
       END DO
       DO J=JMINO,JMAXO
       DO I=IMINO,IMAXO
       IF(DEPFIX(I,J).NE.0.) DEPCC(I,J)=DEPFIX(I,J)
       XN(I,J)=DEPCC(I,J)
       END DO
       END DO
C
c     DO J=10,63
c     XN(6,J)= 2.5
c     XN(7,J)= 2.5
c     XN(8,J)= 2.5
c     XN(9,J)= 2.5
c     END DO
c     END DO
C
c **  do multiple smoothing passes without fixing
C
c      DO ND=1,2 
c      DO J=JMINO,JMAXO
c      DO I=IMINO,IMAXO
c      IF(IJCT(I,J).GE.1.AND.IJCT(I,J).LE.5) THEN
c        CDTMP=1./( RKII(I+1,J)+RKII(I,J)+RKJJ(I,J+1)+RKJJ(I,J) )
c        DEPCC(I,J)=CDTMP*(RKII(I+1,J)*XN(I+1,J)+RKII(I,J)*XN(I-1,J)
c    $                   +RKJJ(I,J+1)*XN(I,J+1)+RKJJ(I,J)*XN(I,J-1))
c      END IF
c      END DO
c      END DO
c      DO J=JMINO,JMAXO
c      DO I=IMINO,IMAXO
c      DEPCC(I,J)=0.875*XN(I,J)+0.125*DEPCC(I,J)
c      XN(I,J)=DEPCC(I,J)
c      END DO
c      END DO
      END DO
C
      END IF
C
C********************************************************************C
C
C     insert any hardwired depths here!
C     KINGS CREEK OPEN BOUNDARY FIX!
C
c     DO J=10,63
c     DEPCC(6,J)= 2.5
c     DEPCC(7,J)= 2.5
c     DEPCC(8,J)= 2.5
c     DEPCC(9,J)= 2.5
c     END DO
c
      DO N=1,NDEPSMF
C
      DO J=JMINO,JMAXO
      DO I=IMINO,IMAXO
      XN(I,J)=DEPCC(I,J)
      END DO
      END DO
C
       DO J=JMINO,JMAXO
       DO I=IMINO,IMAXO
        IF(IJCT(I,J).GE.1.AND.IJCT(I,J).LE.5) THEN
        CTOT=RKII(I+1,J)+RKII(I,J)+RKJJ(I,J+1)+RKJJ(I,J)
		CCEN=(1.-0.0625*CTOT)*XN(I,J)
        CAVG=0.0625*(RKII(I+1,J)*XN(I+1,J)+RKII(I,J)*XN(I-1,J)
     $                   +RKJJ(I,J+1)*XN(I,J+1)+RKJJ(I,J)*XN(I,J-1))
        DEPCC(I,J)=CCEN+CAVG
       END IF
       END DO
       END DO
C
      END DO
C
c     DO J=10,63
c     DEPCC(6,J)=-3.65
c     DEPCC(45,J)=-3.05
c     DEPCC(6,J)= 2.5
c     DEPCC(7,J)= 2.5
c     DEPCC(8,J)= 2.5
c     END DO
c
C********************************************************************C
C
C     OUTPUT NONZERO GRID COORDINATES
C
      OPEN (39,FILE='gridext.out',STATUS='UNKNOWN')
C
      DO J=JMINO,JMAXO
      DO I=IMINO,IMAXO
      IF(X(I,J).NE.0.0.AND.Y(I,J).NE.0.0) THEN 
        WRITE(39,7320)I,J,X(I,J),Y(I,J)
      END IF
      END DO
      END DO
C
      CLOSE(39)
C
 7320 FORMAT(2I5,2X,F10.6,2X,F10.6)
C
C********************************************************************C
C
      OPEN(30,FILE='salt.inp',STATUS='UNKNOWN')
C
      WRITE(14,1411)
      WRITE(14,1412)
      WRITE(14,1413)
      WRITE(14,1414)
      WRITE(16,1611)
      WRITE(16,1612)
      WRITE(16,1613)
      WRITE(16,1614)
      WRITE(15,1500)
C
      N999=0
      DEPMAX=0. 
      NWCELLS=0.
      ASQRTG=0.
      AHIHJ=0.
      DO J=JMINO,JMAXO-1
      DO I=IMINO,IMAXO-1
      IF (IJCT(I,J).GE.1.AND.IJCT(I,J).LE.5) THEN
        NWCELLS=NWCELLS+1
        DXDI=0.5*(X(I+1,J+1)-X(I,J+1)+X(I+1,J)-X(I,J))
        DYDI=0.5*(Y(I+1,J+1)-Y(I,J+1)+Y(I+1,J)-Y(I,J))
        DXDJ=0.5*(X(I+1,J+1)-X(I+1,J)+X(I,J+1)-X(I,J))
        DYDJ=0.5*(Y(I+1,J+1)-Y(I+1,J)+Y(I,J+1)-Y(I,J))
        GII=DXDI*DXDI+DYDI*DYDI
        GJJ=DXDJ*DXDJ+DYDJ*DYDJ
        GIJ=DXDI*DXDJ+DYDI*DYDJ
        ANGD=(ACOS(GIJ/SQRT(GII*GJJ)))*57.29578
        ANGERR=ABS(90.-ANGD)
        HII=SQRT(GII)
        HJJ=SQRT(GJJ)
        CEU=DXDI/HII
        CEV=DXDJ/HJJ
        CNU=DYDI/HII
        CNV=DYDJ/HJJ
        XLNUTME=0.25*(X(I,J)+X(I+1,J)+X(I+1,J+1)+X(I,J+1))
        YLTUTMN=0.25*(Y(I,J)+Y(I+1,J)+Y(I+1,J+1)+Y(I,J+1))
        IF(ISIDEP.EQ.1) THEN
          IF(ISIDPTYP.EQ.1) THEN
            DEPTH=DEPCC(I,J)       
            BELV=-1.*DEPTH
            ZROUGH=ROUGH
            DEPMAX=MAX(DEPMAX,DEPTH)
           ELSE
            BELV=DEPCC(I,J)
            DEPTH=SURFELV-BELV
c fix for okee
            DEPTH=MAX(DEPMIN,DEPTH)
c end fix
            ZROUGH=ROUGH
            DEPMAX=MAX(DEPMAX,DEPTH)
          END IF 
         ELSE
          DEPTH=0.
          BELV=0.0
          ZROUGH=ROUGH
        END IF
	  wndshe=1.0
        HIIHJJ=HII*HJJ
        AHIHJ=AHIHJ+HIIHJJ
        SQRTG=SQRT(GII*GJJ-GIJ*GIJ)
        ASQRTG=ASQRTG+SQRTG
        HII=HII*HSCALE
        HJJ=HJJ*HSCALE
        WRITE(14,1400)I,J,HII,HJJ,DEPTH,BELV,ZROUGH,NVEGIJ(I,J)
        WRITE(6,1400)I,J,HII,HJJ,DEPTH,BELV,ZROUGH,NVEGIJ(I,J)
        WRITE(66,1400)I,J,HII,HJJ,DEPTH,BELV,ZROUGH,NVEGIJ(I,J)
        WRITE(15,1401)I,J,HII,HJJ,HIIHJJ,SQRTG,ANGERR
        WRITE(16,1600)I,J,XLNUTME,YLTUTMN,CEU,CEV,CNU,CNV,wndshe
c hardwire saltmp for shelf here
        SALTMP=0.0
        WRITE(30,3131)NWCELLS,I,J,SALTMP,SALTMP,SALTMP,SALTMP,
     $                SALTMP,SALTMP,SALTMP,SALTMP,SALTMP,SALTMP
C **  BEGIN DXF WRITE SEQUENCE TO FILE grid.dxf
C       The following section added by M.R. Morton, Tetra Tech:
C       Write coordinates of each cell corner to DXF format file.
C       Each cell will be a closed polyline in the DXF file:
C       IJCT = 1, triangular computational cell in NE quadrant: 
        if (IJCT(I,J) .eq. 1) then                     
          write(27,1654) X(I  ,J  ),Y(I  ,J  )         
          write(27,1655) X(I+1,J  ),Y(I+1,J  )         
          write(27,1655) X(I  ,J+1),Y(I  ,J+1)         
          write(27,1655) X(I  ,J  ),Y(I  ,J  )         
        end if                                         
C       IJCT = 4, computational cell in SE quadrant:       
        if (IJCT(I,J) .eq. 4) then                     
          write(27,1654) X(I  ,J  ),Y(I  ,J  )         
          write(27,1655) X(I+1,J  ),Y(I+1,J  )         
          write(27,1655) X(I+1,J+1),Y(I+1,J+1)         
          write(27,1655) X(I  ,J  ),Y(I  ,J  )         
        end if                                         
C       IJCT = 3, computational cell in SW quadrant:       
        if (IJCT(I,J) .eq. 3) then                     
          write(27,1654) X(I+1,J  ),Y(I+1,J  )         
          write(27,1655) X(I+1,J+1),Y(I+1,J+1)         
          write(27,1655) X(I  ,J+1),Y(I  ,J+1)         
          write(27,1655) X(I+1,J  ),Y(I+1,J  )         
        end if                                         
C       IJCT = 2, computational cell in NW quadrant:       
        if (IJCT(I,J) .eq. 2) then                     
          write(27,1654) X(I  ,J  ),Y(I  ,J  )         
          write(27,1655) X(I+1,J+1),Y(I+1,J+1)         
          write(27,1655) X(I  ,J+1),Y(I  ,J+1)         
          write(27,1655) X(I  ,J  ),Y(I  ,J  )         
        end if                                         
C       IJCT = 5, normal four-sided water cell:            
        if (IJCT(I,J) .eq. 5) then                     
          write(27,1654) X(I  ,J  ),Y(I  ,J  )         
          write(27,1655) X(I+1,J  ),Y(I+1,J  )         
          write(27,1655) X(I+1,J+1),Y(I+1,J+1)         
          write(27,1655) X(I  ,J+1),Y(I  ,J+1)         
          write(27,1655) X(I  ,J  ),Y(I  ,J  )         
        end if
C       IJCT = 1, triangular computational cell in NE quadrant: 
        if (IJCT(I,J) .eq. 1) then                     
          write(25,1650)                               
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1651) X(I+1,J  ),Y(I+1,J  )         
          write(25,1651) X(I  ,J+1),Y(I  ,J+1)         
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1652)                               
        end if                                         
C       IJCT = 4, computational cell in SE quadrant:       
        if (IJCT(I,J) .eq. 4) then                     
          write(25,1650)                               
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1651) X(I+1,J  ),Y(I+1,J  )         
          write(25,1651) X(I+1,J+1),Y(I+1,J+1)         
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1652)                               
        end if                                         
C       IJCT = 3, computational cell in SW quadrant:       
        if (IJCT(I,J) .eq. 3) then                     
          write(25,1650)                               
          write(25,1651) X(I+1,J  ),Y(I+1,J  )         
          write(25,1651) X(I+1,J+1),Y(I+1,J+1)         
          write(25,1651) X(I  ,J+1),Y(I  ,J+1)         
          write(25,1651) X(I+1,J  ),Y(I+1,J  )         
          write(25,1652)                               
        end if                                         
C       IJCT = 2, computational cell in NW quadrant:       
        if (IJCT(I,J) .eq. 2) then                     
          write(25,1650)                               
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1651) X(I+1,J+1),Y(I+1,J+1)         
          write(25,1651) X(I  ,J+1),Y(I  ,J+1)         
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1652)                               
        end if                                         
C       IJCT = 5, normal four-sided water cell:            
        if (IJCT(I,J) .eq. 5) then                     
          write(25,1650)                               
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1651) X(I+1,J  ),Y(I+1,J  )         
          write(25,1651) X(I+1,J+1),Y(I+1,J+1)         
          write(25,1651) X(I  ,J+1),Y(I  ,J+1)         
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1652)                               
        end if
C **  END DXF WRITE SEQUENCE TO FILE grid.dxf   
        ICOMP(NWCELLS)=I
        JCOMP(NWCELLS)=J
        XCOMP(NWCELLS)=XLNUTME
        YCOMP(NWCELLS)=YLTUTMN
      END IF
      END DO
      END DO
C 
C **  WRITE CLOSURE TO DXF FILE gird.dxf:                         
C     WRITE(25,1653) MOVED TO JUST BEFORE CLOSE(25)                                            
C
1650  FORMAT('  0',/,'POLYLINE',/,'  8',/,'GRID',/,' 66',/,'     1')
1651  FORMAT('  0',/,'VERTEX',/,'  8',/,'GRID',/,' 10',/,F12.3,/,   
     +       ' 20',/,F12.3)                          
1652  FORMAT('  0',/,'SEQEND')                       
1653  FORMAT('  8',/,'GRID',/,'  0',/,'ENDSEC',/,'  0',/,'EOF') 
1654  FORMAT(2F12.3,'  -1')
1655  FORMAT(2F12.3,'   1')
C
      AERR=2.*ABS(ASQRTG-AHIHJ)/(ASQRTG+AHIHJ)
      WRITE(15,1501)
      WRITE(15,1502)ASQRTG,AHIHJ,AERR
      WRITE(15,1503)NWCELLS
      WRITE(6,1503)NWCELLS
      WRITE(66,1503)NWCELLS
      WRITE(6,1509)N999
      WRITE(66,1509)N999
C
      CLOSE(30)
      IF (ISGG.EQ.0) GO TO 9000
C
C--------------------------------------------------------------------C
C
C **  PROCESS GRAPHICS GRID OVERLAY
C
      OPEN(17,FILE='gcell.inp',STATUS='UNKNOWN')
      OPEN(18,FILE='gcellmap.out',STATUS='UNKNOWN')
      WRITE(18,1811)
      WRITE(18,1812)
      WRITE(18,1813)
      WRITE(18,1814)
C
C **  SKIP OVER TITLE AND AND HEADER LINES AND DETERMINE FILE FORMAT
C 
      DO IS=1,4
      READ(17,1111)
      END DO
C
      READ(17,3003)JCTMP
      CLOSE(17)
      OPEN (17,FILE='gcell.inp',STATUS='UNKNOWN')
C
C **  SKIP OVER TITLE AND AND HEADER LINES
C 
      DO IS=1,4
      READ(17,1111)
      END DO
C
      IF(JCTMP.NE.JGM) THEN
C **    READ OLD FILE FORMAT
        DO JT=1,JGM,120
        JF=JT
        JLAST=JT+119
        IF(JLAST.GT.JGM) JLAST=JGM
        WRITE (7,707)JF,JLAST 
        DO I=1,IGM
        READ (17,3001) (IJCTG(I,J),J=JF,JLAST)
        WRITE(7,701)I,(IJCTG(I,J),J=JF,JLAST) 
        END DO
        END DO
       ELSE
C **    READ NEW FILE FORMAT
C       READ(17,3003)JCTMP
        DO IT=1,IGM,120
        IFIRST=IT
        ILAST=IT+119
        IF(ILAST.GT.IGM) ILAST=IGM
        WRITE (7,7007)IFIRST,ILAST
        DO J=JGM,1,-1
        READ (17,3003)JDUMY,(IJCTG(I,J),I=IFIRST,ILAST)
        WRITE (7,701)JDUMY,(IJCTG(I,J),I=IFIRST,ILAST)
        END DO
        END DO
      END IF
C
      NWGG=0
      WRITE(18,188)IGM,JGM,NWTGG
      DO JG=1,JGM
      WRITE(6,6998)JG
      WRITE(66,6998)JG
      DO IG=1,IGM
      IF (IJCTG(IG,JG).NE.0) THEN
        NWGG=NWGG+1
        WRITE(6,6999)NWGG  
        WRITE(66,6999)NWGG     
        XGG=CDLON1+(CDLON2*FLOAT(IG)+CDLON3)/60.
        YGG=CDLAT1+(CDLAT2*FLOAT(JG)+CDLAT3)/60.
C **    FIND THE CLOSEST COMP CELL
        IF(NWTGG.GE.1) THEN
        RMIN1=1.E+9
        DO NW=1,NWCELLS
        RTMP=SQRT( (XCOMP(NW)-XGG)**2 +(YCOMP(NW)-YGG)**2 )
        IF (RTMP.LT.RMIN1) THEN
          NW1=NW
          RMIN1=RTMP
          ICOMPT1=ICOMP(NW)
          JCOMPT1=JCOMP(NW)
        END IF
        END DO
        END IF
C **    FIND THE 2ND CLOSEST COMP CELL
        IF(NWTGG.GE.2) THEN
        RMIN2=1.E+9
        DO NW=1,NWCELLS
        IF(NW.NE.NW1) THEN
          RTMP=SQRT( (XCOMP(NW)-XGG)**2 +(YCOMP(NW)-YGG)**2 )
          IF (RTMP.LT.RMIN2) THEN
            NW2=NW
            RMIN2=RTMP
            ICOMPT2=ICOMP(NW)
            JCOMPT2=JCOMP(NW)
          END IF
        END IF
        END DO
        END IF
C **    FIND THE 3RD CLOSEST COMP CELL
        IF(NWTGG.GE.3) THEN
        RMIN3=1.E+9
        DO NW=1,NWCELLS
        IF(NW.NE.NW1.AND.NW.NE.NW2) THEN
          RTMP=SQRT( (XCOMP(NW)-XGG)**2 +(YCOMP(NW)-YGG)**2 )
          IF (RTMP.LT.RMIN3) THEN
            NW3=NW
            RMIN3=RTMP
            ICOMPT3=ICOMP(NW)
            JCOMPT3=JCOMP(NW)
          END IF
        END IF
        END DO
        END IF
C **    FIND THE 4TH CLOSEST COMP CELL
        IF(NWTGG.GE.4) THEN
        RMIN4=1.E+9
        DO NW=1,NWCELLS
        IF(NW.NE.NW1.AND.NW.NE.NW2.AND.NW.NE.NW3) THEN
          RTMP=SQRT( (XCOMP(NW)-XGG)**2 +(YCOMP(NW)-YGG)**2 )
          IF (RTMP.LT.RMIN4) THEN
            NW4=NW
            RMIN4=RTMP
            ICOMPT4=ICOMP(NW)
            JCOMPT4=JCOMP(NW)
          END IF
        END IF
        END DO
        END IF
C **  INSERT A BILINEAR INTERPOLATION TO DETERMINE FOUR 
C **  SETS OF COMP CELL INDICES AND 
        IF(NWTGG.GE.1) 
     $    WRITE(18,188)IG,JG,ICOMPT1,JCOMPT1,RMIN1
        IF(NWTGG.GE.2) 
     $    WRITE(18,188)IG,JG,ICOMPT2,JCOMPT2,RMIN2
        IF(NWTGG.GE.3) 
     $    WRITE(18,188)IG,JG,ICOMPT3,JCOMPT3,RMIN3
        IF(NWTGG.GE.4) 
     $    WRITE(18,188)IG,JG,ICOMPT4,JCOMPT4,RMIN4
      END IF
      END DO
      END DO
C
      WRITE(18,188)NWGG
      CLOSE(17)
      CLOSE(18)
 6998 FORMAT(' JG = ',I5)
 6999 FORMAT(' NWGG = ',I5)
C       
      GO TO 9000
C
C--------------------------------------------------------------------C
C
 1400 FORMAT(1X,I5,2X,I5,2X,E12.5,2X,E12.5,2X,E12.5,2X,E12.5,
     $       2X,E12.5,2X,I3)
 1401 FORMAT(1X,I5,1X,I5,5(2X,E11.4))
 1500 FORMAT(5X,'I',5X,'J',4X,'HII',10X,'HJJ',10X,'HIIHJJ',6X,
     $       'JACOBIAN',5X,'ANG ERROR',/)            
 1501 FORMAT(//)
 1502 FORMAT(1X,'ASQRTG=',2X,E11.4,2X,'ASHIHJ=',2X,E11.4,2X,'AERR=',
     $       2X,E11.4)
 1503 FORMAT(1X,'NWCELLS=',I10)
 1509 FORMAT(1X,'N999 =',I10)
 1600 FORMAT(1X,I5,1X,I5,7(1X,E13.6))
  188 FORMAT(4I10,5X,F12.4)
 1411 FORMAT('C dxdy.inp file, in free format across columns')
 1412 FORMAT('C')
 1413 FORMAT('C     I     J        DX            DY           ',
     $       1X,'DEPTH     BOTTOM ELEV      ZROUGH  VEG TYPE')
 1414 FORMAT('C')
 1611 FORMAT('C lxly.inp file, in free format across line')
 1612 FORMAT('C')
 1613 FORMAT('C    I     J    XLNUTME       YLTUTMN        CCUE',
     $        1X,'           CCVE          CCUN         CCVN')  
 1614 FORMAT('C')
 1811 FORMAT('C gcellmap.inp file, in free format across columns')
 1812 FORMAT('C')
 1813 FORMAT('  IGRAPHIC  JGRAPHIC     ICOMP     JCOMP         RMIN')
 1814 FORMAT('C')
 6777 FORMAT('NO DEP DATA AT I,J,X,Y = ',2I5,2E14.5)
 6779 FORMAT('DEPTH TOO LARGE AT AT I,J,DEPTH = ',2I5,2E14.5)
 6888 FORMAT('NO VEG DATA AT I,J,X,Y = ',2I5,2E14.5)
 6778 FORMAT(2I5,2X,F10.4,2X,F10.4,2X,F12.4,2X,F12.4)
 3131 FORMAT(3I5,2X,12F6.1)
 6769 FORMAT('SMOOTH ERR, I,J,CTOT=',2I5,E14.5)
C
C********************************************************************C
C
C     CARTESIAN GRID DEPTH INTERPOLATION AND dxdy.inp and lxly.inp
C     FILE GENERARTION NTYPE = 8 OR 9
C
 8000 CONTINUE
C
C--------------------------------------------------------------------C
C
C **  READ CELL TYPES FROM FILE cell.inp
C
C **  SKIP OVER TITLE AND AND HEADER LINES AND DETERMINE FILE FORMAT
C 
      DO IS=1,4
      READ(3,1111)
      END DO
      READ(3,3003)JCTMP
      CLOSE(3)
      OPEN (3,FILE='CELL.INP',STATUS='UNKNOWN')
C
C **  SKIP OVER TITLE AND AND HEADER LINES
C 
      DO IS=1,4
      READ(3,1111)
      END DO
C
      IF(JCTMP.NE.JC) THEN
C **    READ OLD FILE FORMAT
        DO JT=1,JC,120
        JF=JT
        JLAST=JT+119
        IF(JLAST.GT.JC) JLAST=JC
        WRITE (7,700)JF,JLAST 
        DO I=1,IC
        READ (3,3001) (IJCT(I,J),J=JF,JLAST)
        WRITE(7,701)I,(IJCT(I,J),J=JF,JLAST)     
        END DO
        END DO
       ELSE
C **    READ NEW FILE FORMAT
C       READ(3,3003)JCTMP
        DO IT=1,IC,120
        IFIRST=IT
        ILAST=IT+119
        IF(ILAST.GT.IC) ILAST=IC
        WRITE (7,7001)IFIRST,ILAST 
        DO J=JC,1,-1
        READ (3,3003)JDUMY,(IJCT(I,J),I=IFIRST,ILAST)
        WRITE (7,701)JDUMY,(IJCT(I,J),I=IFIRST,ILAST)
        END DO
        END DO
      END IF
C
C--------------------------------------------------------------------C
C
      OPEN(18,FILE='gcellmap.inp',STATUS='UNKNOWN')
      OPEN(30,FILE='salt.inp',STATUS='UNKNOWN')
C
      NWTGG=1
      WRITE(14,1411)
      WRITE(14,1412)
      WRITE(14,1413)
      WRITE(14,1414)
      WRITE(16,1611)
      WRITE(16,1612)
      WRITE(16,1613)
      WRITE(16,1614)
      WRITE(18,1811)
      WRITE(18,1812)
      WRITE(18,1813)
      WRITE(18,1814)
      WRITE(18,188)IGM,JGM,NWTGG
C
      IF(NTYPE.NE.9) THEN
C
      N999=0
      DEPMAX=0.
      NWCELLS=0.
      DO J=1,JC
      DO I=1,IC
      IF (IJCT(I,J).GE.1.AND.IJCT(I,J).LE.5) THEN
        NWCELLS=NWCELLS+1        
        CEU=1.0
        CEV=0.0
        CNU=0.0
        CNV=1.0
        XLNUTME=CDLON1+(CDLON2*FLOAT(I)+CDLON3)/60.
        YLTUTMN=CDLAT1+(CDLAT2*FLOAT(J)+CDLAT3)/60.
        RADSQ1=DXCG*DYCG
        RADSQ2=DXCG*DYCG
        IF(ISIDEP.EQ.1) THEN
          CALL RADDEP(I,J,XLNUTME,YLTUTMN,RADSQ1,RADSQ2,DEPTH)
          IF(DEPTH.EQ.-999.) THEN
		    N999=N999+1
            WRITE(67,6777)I,J,XLNUTME,YLNUTMN
          END IF
          IF(ISIDPTYP.EQ.1) THEN       
            BELV=-1.*DEPTH
            ZROUGH=ROUGH
            NVEGDUM=0
            DEPMAX=MAX(DEPMAX,DEPTH)
           ELSE
            BELV=DEPTH
            DEPTH=SURFELV-BELV
            ZROUGH=ROUGH
            NVEGDUM=0
            DEPMAX=MAX(DEPMAX,DEPTH)
          END IF 
         ELSE
          DEPTH=0.
          BELV=0.0
          ZROUGH=ROUGH
          NVEGDUM=0
        END IF
	  	  wndshe=1.0
   
        WRITE(14,1400)I,J,DXCG,DYCG,DEPTH,BELV,ZROUGH,NVEGDUM
        WRITE(6,1400)I,J,DXCG,DYCG,DEPTH,BELV,ZROUGH,NVEGDUM
        WRITE(66,1400)I,J,DXCG,DYCG,DEPTH,BELV,ZROUGH,NVEGDUM
        WRITE(16,1600)I,J,XLNUTME,YLTUTMN,CEU,CEV,CNU,CNV,wndshe
C **  BEGIN DXF WRITE SEQUENCE TO FILE grid.dxf
C       The following section added by M.R. Morton, Tetra Tech:
C       Write coordinates of each cell corner to DXF format file.
C       Each cell will be a closed polyline in the DXF file:
C       IJCT = 1, triangular computational cell in NE quadrant: 
C       IJCT = 1, triangular computational cell in NE quadrant: 
        if (IJCT(I,J) .eq. 1) then                     
          write(27,1654) X(I  ,J  ),Y(I  ,J  )         
          write(27,1655) X(I+1,J  ),Y(I+1,J  )         
          write(27,1655) X(I  ,J+1),Y(I  ,J+1)         
          write(27,1655) X(I  ,J  ),Y(I  ,J  )         
        end if                                         
C       IJCT = 4, computational cell in SE quadrant:       
        if (IJCT(I,J) .eq. 4) then                     
          write(27,1654) X(I  ,J  ),Y(I  ,J  )         
          write(27,1655) X(I+1,J  ),Y(I+1,J  )         
          write(27,1655) X(I+1,J+1),Y(I+1,J+1)         
          write(27,1655) X(I  ,J  ),Y(I  ,J  )         
        end if                                         
C       IJCT = 3, computational cell in SW quadrant:       
        if (IJCT(I,J) .eq. 3) then                     
          write(27,1654) X(I+1,J  ),Y(I+1,J  )         
          write(27,1655) X(I+1,J+1),Y(I+1,J+1)         
          write(27,1655) X(I  ,J+1),Y(I  ,J+1)         
          write(27,1655) X(I+1,J  ),Y(I+1,J  )         
        end if                                         
C       IJCT = 2, computational cell in NW quadrant:       
        if (IJCT(I,J) .eq. 2) then                     
          write(27,1654) X(I  ,J  ),Y(I  ,J  )         
          write(27,1655) X(I+1,J+1),Y(I+1,J+1)         
          write(27,1655) X(I  ,J+1),Y(I  ,J+1)         
          write(27,1655) X(I  ,J  ),Y(I  ,J  )         
        end if                                         
C       IJCT = 5, normal four-sided water cell:            
        if (IJCT(I,J) .eq. 5) then                     
          write(27,1654) X(I  ,J  ),Y(I  ,J  )         
          write(27,1655) X(I+1,J  ),Y(I+1,J  )         
          write(27,1655) X(I+1,J+1),Y(I+1,J+1)         
          write(27,1655) X(I  ,J+1),Y(I  ,J+1)         
          write(27,1655) X(I  ,J  ),Y(I  ,J  )         
        end if
        if (IJCT(I,J) .eq. 1) then                     
          write(25,1650)                               
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1651) X(I+1,J  ),Y(I+1,J  )         
          write(25,1651) X(I  ,J+1),Y(I  ,J+1)         
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1652)                               
        end if                                         
C       IJCT = 4, computational cell in SE quadrant:       
        if (IJCT(I,J) .eq. 4) then                     
          write(25,1650)                               
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1651) X(I+1,J  ),Y(I+1,J  )         
          write(25,1651) X(I+1,J+1),Y(I+1,J+1)         
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1652)                               
        end if                                         
C       IJCT = 3, computational cell in SW quadrant:       
        if (IJCT(I,J) .eq. 3) then                     
          write(25,1650)                               
          write(25,1651) X(I+1,J  ),Y(I+1,J  )         
          write(25,1651) X(I+1,J+1),Y(I+1,J+1)         
          write(25,1651) X(I  ,J+1),Y(I  ,J+1)         
          write(25,1651) X(I+1,J  ),Y(I+1,J  )         
          write(25,1652)                               
        end if                                         
C       IJCT = 2, computational cell in NW quadrant:       
        if (IJCT(I,J) .eq. 2) then                     
          write(25,1650)                               
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1651) X(I+1,J+1),Y(I+1,J+1)         
          write(25,1651) X(I  ,J+1),Y(I  ,J+1)         
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1652)                               
        end if                                         
C       IJCT = 5, normal four-sided water cell:            
        if (IJCT(I,J) .eq. 5) then                     
          write(25,1650)                               
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1651) X(I+1,J  ),Y(I+1,J  )         
          write(25,1651) X(I+1,J+1),Y(I+1,J+1)         
          write(25,1651) X(I  ,J+1),Y(I  ,J+1)         
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1652)                               
        end if
C **  END DXF WRITE SEQUENCE TO FILE grid.dxf   
        ICOMPT=I
        JCOMPT=J
        RMIN=0.
        WRITE(18,188)I,J,ICOMPT,JCOMPT,RMIN
      END IF
      END DO
      END DO
C
      ELSE
C
      CALL VUTMBAY
c
      DO J=1,JC
      DO I=1,IC
      XLNUTME=ABS(CDLON1+(CDLON2*FLOAT(I)+CDLON3)/60.)
      YLTUTMN=CDLAT1+(CDLAT2*FLOAT(J)+CDLAT3)/60.
c next line for chesbay con shelf only
c     YLTUTMN=CDLAT1+(CDLAT2*FLOAT(J-100)+CDLAT3)/60.
      XCELL(I,J)=XUTMBAY(XLNUTME,YLTUTMN)
      YCELL(I,J)=YUTMBAY(XLNUTME,YLTUTMN)
      XCELL(I,J)=XCELL(I,J)*1.E-3
      YCELL(I,J)=(YCELL(I,J)-0.4E+7)*1.E-3
      END DO
      END DO
C
      DO J=2,JC
      DO I=2,IC
      IF(IJCT(I,J).EQ.0) THEN
        X(I,J)=0.
        Y(I,J)=0.
      ELSE
        X(I,J)=0.25*(XCELL(I,J)+XCELL(I-1,J)
     $            +XCELL(I,J-1)+XCELL(I-1,J-1))
        Y(I,J)=0.25*(YCELL(I,J)+YCELL(I-1,J)
     $            +YCELL(I,J-1)+YCELL(I-1,J-1))
      END IF
      END DO
      END DO
c
      N999=0
      DEPMAX=0.
      NWCELLS=0
      DO J=1,JC
      DO I=1,IC
      XLNUTME=ABS(CDLON1+(CDLON2*FLOAT(I)+CDLON3)/60.)
      YLTUTMN=CDLAT1+(CDLAT2*FLOAT(J)+CDLAT3)/60.
c next line for bay and con shelf
c     YLTUTMN=CDLAT1+(CDLAT2*FLOAT(J-100)+CDLAT3)/60.
      DLONDD(I,J)=XLNUTME
      DLATDD(I,J)=YLTUTMN      
      XCELL(I,J)=XUTMBAY(XLNUTME,YLTUTMN)
      YCELL(I,J)=YUTMBAY(XLNUTME,YLTUTMN)
      IF (IJCT(I,J).GE.1.AND.IJCT(I,J).LE.5) NWCELLS=NWCELLS+1
      END DO
      END DO
      DO J=1,JC
      DO I=1,IC
      IF (IJCT(I,J).GE.1.AND.IJCT(I,J).LE.5) THEN
C         IF(I.LT.42) THEN
C           SALTMP=0.
C          ELSE
C           IF(J.LE.25) SALTMP=32.
C           IF(J.GT.25.AND.J.LT.149) THEN
C             SALTMP=32.*(150-FLOAT(J))/125.
C           END IF
C           IF(J.GE.150) SALTMP=0.
C         END IF
C       WRITE(30,3131)NWCELLS,I,J,SALTMP,SALTMP,SALTMP,SALTMP,
C    $                SALTMP,SALTMP,SALTMP,SALTMP,
C    $                SALTMP,SALTMP,SALTMP,SALTMP
        CEU=1.0
        CEV=0.0
        CNU=0.0
        CNV=1.0
        DXCCTR=0.5*ABS(XCELL(I+1,J)-XCELL(I-1,J))
        DYCCTR=0.5*ABS(YCELL(I,J+1)-YCELL(I,J-1))
C       XLNUTME=XCELL(I,J)
C       YLTUTMN=YCELL(I,J)
C       RADSQ=DXCCTR*DYCCTR
C       corrections for ches bay local utm system
        XLNUTME=XCELL(I,J)*1.E-3
        YLTUTMN=(YCELL(I,J)-0.4E+7)*1.E-3
        RADSQ1=DXCCTR*DYCCTR*1.E-6
        RADSQ2=DXCCTR*DYCCTR*1.E-6
        IF(ISIDEP.EQ.1) THEN
          CALL RADDEP(I,J,XLNUTME,YLTUTMN,RADSQ1,RADSQ2,DEPTH)
          IF(DEPTH.EQ.-999.) THEN
            WRITE(67,6778)I,J,DLONDD(I,J),DLATDD(I,J),
     $                        XLNUTME,YLTUTMN 
          END IF
          IF(DEPTH.LT.1.0) DEPTH=1.0
          BELV=-1.*DEPTH
          ZROUGH=ROUGH
          NVEGDUM=0
          DEPMAX=MAX(DEPMAX,DEPTH)
         ELSE
          DEPTH=0.
          BELV=0.0
          ZROUGH=ROUGH
          NVEGDUM=0
        END IF
        wndshe=1.0
	  WRITE(14,1400)I,J,DXCCTR,DYCCTR,DEPTH,BELV,ZROUGH,NVEGDUM
        WRITE(6,1400)I,J,DXCCTR,DYCCTR,DEPTH,BELV,ZROUGH,NVEGDUM
        WRITE(66,1400)I,J,DXCCTR,DYCCTR,DEPTH,BELV,ZROUGH,NVEGDUM
        WRITE(16,1600)I,J,XLNUTME,YLTUTMN,CEU,CEV,CNU,CNV,wndshe
C **  BEGIN DXF WRITE SEQUENCE TO FILE grid.dxf
C       The following section added by M.R. Morton, Tetra Tech:
C       Write coordinates of each cell corner to DXF format file.
C       Each cell will be a closed polyline in the DXF file:
C       IJCT = 1, triangular computational cell in NE quadrant: 
C       IJCT = 1, triangular computational cell in NE quadrant: 
        if (IJCT(I,J) .eq. 1) then                     
          write(27,1654) X(I  ,J  ),Y(I  ,J  )         
          write(27,1655) X(I+1,J  ),Y(I+1,J  )         
          write(27,1655) X(I  ,J+1),Y(I  ,J+1)         
          write(27,1655) X(I  ,J  ),Y(I  ,J  )         
        end if                                         
C       IJCT = 4, computational cell in SE quadrant:       
        if (IJCT(I,J) .eq. 4) then                     
          write(27,1654) X(I  ,J  ),Y(I  ,J  )         
          write(27,1655) X(I+1,J  ),Y(I+1,J  )         
          write(27,1655) X(I+1,J+1),Y(I+1,J+1)         
          write(27,1655) X(I  ,J  ),Y(I  ,J  )         
        end if                                         
C       IJCT = 3, computational cell in SW quadrant:       
        if (IJCT(I,J) .eq. 3) then                     
          write(27,1654) X(I+1,J  ),Y(I+1,J  )         
          write(27,1655) X(I+1,J+1),Y(I+1,J+1)         
          write(27,1655) X(I  ,J+1),Y(I  ,J+1)         
          write(27,1655) X(I+1,J  ),Y(I+1,J  )         
        end if                                         
C       IJCT = 2, computational cell in NW quadrant:       
        if (IJCT(I,J) .eq. 2) then                     
          write(27,1654) X(I  ,J  ),Y(I  ,J  )         
          write(27,1655) X(I+1,J+1),Y(I+1,J+1)         
          write(27,1655) X(I  ,J+1),Y(I  ,J+1)         
          write(27,1655) X(I  ,J  ),Y(I  ,J  )         
        end if                                         
C       IJCT = 5, normal four-sided water cell:            
        if (IJCT(I,J) .eq. 5) then                     
          write(27,1654) X(I  ,J  ),Y(I  ,J  )         
          write(27,1655) X(I+1,J  ),Y(I+1,J  )         
          write(27,1655) X(I+1,J+1),Y(I+1,J+1)         
          write(27,1655) X(I  ,J+1),Y(I  ,J+1)         
          write(27,1655) X(I  ,J  ),Y(I  ,J  )         
        end if
        if (IJCT(I,J) .eq. 1) then                     
          write(25,1650)                               
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1651) X(I+1,J  ),Y(I+1,J  )         
          write(25,1651) X(I  ,J+1),Y(I  ,J+1)         
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1652)                               
        end if                                         
C       IJCT = 4, computational cell in SE quadrant:       
        if (IJCT(I,J) .eq. 4) then                     
          write(25,1650)                               
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1651) X(I+1,J  ),Y(I+1,J  )         
          write(25,1651) X(I+1,J+1),Y(I+1,J+1)         
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1652)                               
        end if                                         
C       IJCT = 3, computational cell in SW quadrant:       
        if (IJCT(I,J) .eq. 3) then                     
          write(25,1650)                               
          write(25,1651) X(I+1,J  ),Y(I+1,J  )         
          write(25,1651) X(I+1,J+1),Y(I+1,J+1)         
          write(25,1651) X(I  ,J+1),Y(I  ,J+1)         
          write(25,1651) X(I+1,J  ),Y(I+1,J  )         
          write(25,1652)                               
        end if                                         
C       IJCT = 2, computational cell in NW quadrant:       
        if (IJCT(I,J) .eq. 2) then                     
          write(25,1650)                               
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1651) X(I+1,J+1),Y(I+1,J+1)         
          write(25,1651) X(I  ,J+1),Y(I  ,J+1)         
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1652)                               
        end if                                         
C       IJCT = 5, normal four-sided water cell:            
        if (IJCT(I,J) .eq. 5) then                     
          write(25,1650)                               
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1651) X(I+1,J  ),Y(I+1,J  )         
          write(25,1651) X(I+1,J+1),Y(I+1,J+1)         
          write(25,1651) X(I  ,J+1),Y(I  ,J+1)         
          write(25,1651) X(I  ,J  ),Y(I  ,J  )         
          write(25,1652)                               
        end if
C **  END DXF WRITE SEQUENCE TO FILE grid.dxf   
        ICOMPT=I
        JCOMPT=J
        RMIN=0.
        WRITE(18,188)I,J,ICOMPT,JCOMPT,RMIN
      END IF
      END DO
      END DO
C
      END IF
C
      WRITE(18,188)NWCELLS
      WRITE(15,1503)NWCELLS
      WRITE(6,1503)NWCELLS
      WRITE(66,1503)NWCELLS
C
      CLOSE(18)
      CLOSE(30)
C
C********************************************************************C
C
C     OUTPUT NONZERO GRID COORDINATES
C
      OPEN (39,FILE='gridext.out',STATUS='UNKNOWN')
C
      DO J=JMINO,JMAXO
      DO I=IMINO,IMAXO
      IF(X(I,J).NE.0.0.AND.Y(I,J).NE.0.0) THEN 
        WRITE(39,7320)I,J,X(I,J),Y(I,J)
      END IF
      END DO
      END DO
C
      CLOSE(39)
C
C********************************************************************C
C
 9000 CONTINUE
      WRITE(6,6006)DEPMAX
      WRITE(66,6006)DEPMAX
 6006 FORMAT(' DEPMAX = ',E12.5)
C
      CLOSE(2)
      CLOSE(3)
      CLOSE(4)
      CLOSE(7)
      CLOSE(8)
      CLOSE(9)
      CLOSE(10)
      CLOSE(11)
      CLOSE(12)
      CLOSE(12)
      CLOSE(14)
      CLOSE(15)
      CLOSE(16)
      CLOSE(66)
      CLOSE(67)
C **  WRITE CLOSURE TO DXF FILE gird.dxf:                         
      WRITE(25,1653)
      WRITE(26,1653)
      CLOSE(25)
      CLOSE(26)
      CLOSE(27)
      CLOSE(89)
C
      STOP
      END
C
C********************************************************************C
C********************************************************************C
C********************************************************************C
C
      SUBROUTINE RADVEG(I,J,UTME,UTMN,RADSQ1,RADSQ2,NVTMP)
C
C********************************************************************C
C
C
      IMPLICIT REAL*8 (A-H,O-Z)
C
      PARAMETER(NDDMAX=8000000,NVDMAX=4000000)
C     PARAMETER(NDDMAX=80000,NVDMAX=2000) ! ches bay
      COMMON /ONE/ DEPD(NDDMAX),DEPE(NDDMAX),DEPN(NDDMAX),CDEP,RADM
      COMMON /TWO/ NDEPDAT
      COMMON /VEG/ NVEGDAT,NVEGTYP,ROUGH,NVEGD(NVDMAX),VEGE(NVDMAX),
     $       VEGN(NVDMAX),NNVEG(0:12)
C
C********************************************************************C
C
      RADSQ=RADM*RADM*RADSQ1
C
      DO NTMP=0,NVEGTYP
      NNVEG(NTMP)=0
      END DO
C
      DO N=1,NVEGDAT
      RSTMP=(VEGE(N)-UTME)**2+(VEGN(N)-UTMN)**2
      IF (RSTMP.LE.RADSQ) THEN
       NTTMP=NVEGD(N)
       NNVEG(NTTMP)=NNVEG(NTTMP)+1
      END IF
      END DO
C
      WRITE(6,600)I,J, (NNVEG(N),N=1,NVEGTYP)
      WRITE(66,600)I,J, (NNVEG(N),N=1,NVEGTYP)
  600 FORMAT(16I5)
C
      NDUM=-1
      DO NTMP=1,NVEGTYP
      IF( NNVEG(NTMP).GT.NDUM) THEN
        NDUM=NNVEG(NTMP)
        NVTMP=NTMP
      END IF
      END DO
      IF(NNVEG(NVTMP).EQ.0) NVTMP=0
C
C     IF(DEPTH.EQ.-999.) THEN
C
C       SUMW=0.
C       SUMWD=0.
C       RADSQ=RADM*RADM*RADSQ2
C
C       DO N=1,NDEPDAT
C       RSTMP=(DEPE(N)-UTME)**2+(DEPN(N)-UTMN)**2
C       IF (RSTMP.LE.RADSQ) THEN
C        WT=(RADSQ-RSTMP)/(RADSQ+RSTMP)
C        WT=WT**CDEP
C        SUMW=SUMW+WT
C        SUMWD=SUMWD+WT*DEPD(N)
C       END IF
C       END DO
C
C       IF(SUMW.EQ.0.)THEN
C         DEPTH=-999.
C        ELSE
C         DEPTH=SUMWD/SUMW
C       END IF
C
C     END IF
C
C********************************************************************C
C
      RETURN
      END
C
C********************************************************************C
C********************************************************************C
C********************************************************************C
C
      SUBROUTINE RADDEP(I,J,UTME,UTMN,RADSQ1,RADSQ2,DEPTH)
C
C********************************************************************C
C
C
      IMPLICIT REAL*8 (A-H,O-Z)
C
      PARAMETER (IMIND=1,JMIND=1,IMAXD=2000,JMAXD=2000)
      PARAMETER(NDDMAX=8000000)
C     PARAMETER(NDDMAX=80000) ! ches bay
      COMMON /ONE/ DEPD(NDDMAX),DEPE(NDDMAX),DEPN(NDDMAX),CDEP,RADM
      COMMON /TWO/ NDEPDAT
      COMMON /CXY/  X(IMIND:IMAXD,JMIND:JMAXD), 
     $              Y(IMIND:IMAXD,JMIND:JMAXD)
C
C********************************************************************C
C
      IF(RADM.LT.0.) GO TO 1000
C
      SUMW=0.
      SUMWD=0.
      RADSQ=RADM*RADM*RADSQ1
C
      DO N=1,NDEPDAT
      RSTMP=(DEPE(N)-UTME)**2+(DEPN(N)-UTMN)**2
      IF (RSTMP.LT.RADSQ) THEN
       WT=(RADSQ-RSTMP)/(RADSQ+RSTMP)
       WT=WT**CDEP
       SUMW=SUMW+WT
       SUMWD=SUMWD+WT*DEPD(N)
      END IF
      END DO
C
      IF(SUMW.EQ.0.)THEN
       DEPTH=-999.
      ELSE
       DEPTH=SUMWD/SUMW
	   ADEP=ABS(DEPTH)
	   IF(ADEP.GT.1000.) THEN
	     WRITE(67,6000)I,J,SUMW,SUMWD,RADSQ,RSTMP
	   END IF
      END IF
C
c      IF(DEPTH.EQ.-999.) THEN
C
c        SUMW=0.
c        SUMWD=0.
c        RADSQ=RADM*RADM*RADSQ2
C
c        DO N=1,NDEPDAT
c        RSTMP=(DEPE(N)-UTME)**2+(DEPN(N)-UTMN)**2
c        IF (RSTMP.LT.RADSQ) THEN
c         WT=(RADSQ-RSTMP)/(RADSQ+RSTMP)
c         WT=WT**CDEP
c         SUMW=SUMW+WT
c         SUMWD=SUMWD+WT*DEPD(N)
c        END IF
c        END DO
C
c        IF(SUMW.EQ.0.)THEN
c          DEPTH=-999.
c         ELSE
c          DEPTH=SUMWD/SUMW
c	      ADEP=ABS(DEPTH)
c	      IF(ADEP.GT.1000.) THEN
c	        WRITE(67,6000)I,J,SUMW,SUMWD,RADSQ,RSTMP
c	      END IF
c        END IF
C
c      END IF
C
      GO TO 2000
C	  
C********************************************************************C
C
 1000 CONTINUE
C
      RADSQ1=RADM*RADM*RADSQ1
      RADSQ2=RADM*RADM*RADSQ2
c
      RADSQ=MAX(RADSQ1,RADSQ2)
C
      XSW=0.
      YSW=0.
      XSE=X(I+1,J)-X(I,J)
      YSE=Y(I+1,J)-Y(I,J) 
      XNE=X(I+1,J+1)-X(I,J)
      YNE=Y(I+1,J+1)-Y(I,J)  
      XNW=X(I,J+1)-X(I,J)
      YNW=Y(I,J+1)-Y(I,J) 
C
      A11=XSE
      A12=YSE
      A13=A11*A12
      A21=XNE
      A22=YNE
      A23=A21*A22
      A31=XNW
      A32=YNW
      A33=A31*A32
C
      DETB=A11*A22*A33+A12*A23*A31+A13*A21*A32
     $    -A31*A22*A13-A32*A23*A11-A33*A21*A12
      DETB=1./DETB
C
      CIXX=(A22*A33+A13*A32-A32*A23-A33*A12)*DETB
C
      CIYY=(A11*A33+A23*A31-A31*A13-A33*A21)*DETB
C
      CIXY=(A12*A31+A21*A32-A31*A22-A32*A11)*DETB
C
      CJXX=(A12*A23+A13*A32-A22*A13-A33*A12)*DETB
C
      CJYY=(A11*A33+A13*A21-A31*A13-A23*A11)*DETB
C
      CJXY=(A11*A22+A12*A31-A32*A11-A21*A12)*DETB
C
      SUMW=0.
      SUMWD=0.
      IP=I+1
      JP=J+1
C
      DEPMXX=-1000.
      DEPMNN=1000.
      DO N=1,NDEPDAT
C
      XVAL=DEPE(N)-X(I,J)
      YVAL=DEPN(N)-Y(I,J)
      IVAL=I+CIXX*XVAL+CIYY*YVAL+CIXY*XVAL*YVAL
      JVAL=J+CJXX*XVAL+CJYY*YVAL+CJXY*XVAL*YVAL
C
       IF(IVAL.GE.I.AND.IVAL.LE.IP) THEN
       IF(JVAL.GE.J.AND.JVAL.LE.JP) THEN
         IF(CDEP.GT.0.001) THEN
           RSTMP=(DEPE(N)-UTME)**2+(DEPN(N)-UTMN)**2
           IF(RSTMP.LT.RADSQ) THEN
             IF(I.GE.12.AND.J.LE.40) THEN
               WRITE(89,8900)I,J,UTME,UTMN,DEPE(N),DEPN(N),DEPD(N)
             END IF
             WT=(RADSQ-RSTMP)/(RADSQ+RSTMP)
             WT=WT**CDEP
             SUMW=SUMW+WT
             SUMWD=SUMWD+WT*DEPD(N)
           END IF
          ELSE
           IF(I.GE.12.AND.J.LE.40) THEN
             WRITE(89,8900)I,J,UTME,UTMN,DEPE(N),DEPN(N),DEPD(N)
           END IF
           WT=1.
           SUMW=SUMW+WT
           SUMWD=SUMWD+WT*DEPD(N)
         END IF 
         DEPMXX=MAX(DEPMXX,DEPD(N))
         DEPMNN=MIN(DEPMNN,DEPD(N))
       END IF
       END IF
C
      END DO
C
      IF(SUMW.EQ.0.)THEN
        DEPTH=-999.
       ELSE
        DEPTH=SUMWD/SUMW
        IF(ISIDPTYP.EQ.2.AND.DEPTH.GT.SURFELV) DEPTH=-999.
        WRITE(89,8901)I,J,DEPMNN,DEPTH,DEPMXX
	    ADEP=ABS(DEPTH)
	    IF(ADEP.GT.1000.) THEN
	       WRITE(67,6000)I,J,SUMW,SUMWD,RADSQ,RSTMP
	    END IF
      END IF
C
 2000 CONTINUE
C
 6000 FORMAT('ERR AT I,J,SUMW,SUMWD,RADSQ,RSTMP=',2I5,4E14.5)
 8900 FORMAT(2I5,4F12.2,F10.2)
 8901 FORMAT('SUMMARY I,J,DMN,DAV,DMX = '2I5,3F10.2)
C
C********************************************************************C
C
      RETURN
      END
C
C********************************************************************C
C********************************************************************C
C********************************************************************C
C
      SUBROUTINE VUTMBAY
      IMPLICIT REAL*8 (A-H,O-Z)
C     REAL*8 PHI, CLONG
      COMMON /CUTM/  RK,PHI0,CMERID, AE,ECC2,ECC2P
C
C********************************************************************C
C
      RK = .9996
      PHI0 = 0.0
      CMERID =75.0
C
      ER = 6378206.4d0
      PR = 6356583.8d0
      AE = ER
      ECC2 = 1. - PR**2./ER**2
      ECC2P = ECC2/(1.-ECC2)
C
C********************************************************************C
C
      RETURN
      END
C
C********************************************************************C
C********************************************************************C
C********************************************************************C
C
      SUBROUTINE CALSMD
C
      IMPLICIT REAL*8 (A-H,O-Z)
C
C
C
C
      PARAMETER (IMIND=1,JMIND=1,IMAXD=2000,JMAXD=2000,IGGDIM=2000,
     $           JGGDIM=2000,
     $           NBPD=40000,NGIM=400000,NWCDIM=4000000)
      PARAMETER(NDDMAX=8000000,NVDMAX=4000000)
C
C ches bay parameters
c     PARAMETER (IMIND=1,JMIND=1,IMAXD=100,JMAXD=200,IGGDIM=100,
c    $           JGGDIM=200,
c    $           NBPD=400,NGIM=2000,NWCDIM=20000)
C     PARAMETER(80000) 
C ches bay parameters
C
C********************************************************************C
C
      COMMON /CSMD/ SMD,IB,IE,JB,JE,IJSMD,ISMD,JSMD
      COMMON /CXY/  X(IMIND:IMAXD,JMIND:JMAXD), 
     $              Y(IMIND:IMAXD,JMIND:JMAXD)
C
C********************************************************************C
C
      IF(IJSMD.NE.0) THEN
        SXTMP=0.5*( X(IE,JB)-X(IB,JB)+X(IE,JE)-X(IB,JE) )
        DO J=JB+1,JE-1
         SXTMP=SXTMP+X(IE,J)-X(IB,J)
        END DO
        SYTMP=0.5*( Y(IB,JE)-Y(IB,JB)+Y(IE,JE)-Y(IE,JB) )
        DO I=IB+1,IE-1
         SYTMP=SYTMP+Y(I,JE)-Y(I,JB)
        END DO
        SMD=SXTMP/SYTMP
        RETURN
      END IF
C
      IF(ISMD.NE.0) THEN
        IF(ISMD.EQ.IB) THEN
          CI  =0.5*(X(ISMD  ,JB)+X(ISMD  ,JE))
          CIP1=0.5*(X(ISMD+1,JB)+X(ISMD+1,JE))
          CIP2=0.5*(X(ISMD+2,JB)+X(ISMD+2,JE))
          DO J=JB+1,JE-1
           CI  =CI  +X(ISMD  ,J)
           CIP1=CIP1+X(ISMD+1,J)
           CIP2=CIP2+X(ISMD+2,J)
          END DO
          SXTMP=-1.5*CI+2.*CIP1-0.5*CIP2
        END IF
        IF(ISMD.GT.IB.AND.ISMD.LT.IE) THEN
          CIM1=0.5*(X(ISMD-1,JB)+X(ISMD-1,JE))
          CIP1=0.5*(X(ISMD+1,JB)+X(ISMD+1,JE))
          DO J=JB+1,JE-1
           CIM1=CIM1+X(ISMD-1,J)
           CIP1=CIP1+X(ISMD+1,J)
          END DO
          SXTMP=0.5*(CIP1-CIM1)
        END IF
        IF(ISMD.EQ.IE) THEN
          CI  =0.5*(X(ISMD  ,JB)+X(ISMD  ,JE))
          CIM1=0.5*(X(ISMD-1,JB)+X(ISMD-1,JE))
          CIM2=0.5*(X(ISMD-2,JB)+X(ISMD-2,JE))
          DO J=JB+1,JE-1
           CI  =CI  +X(ISMD  ,J)
           CIM1=CIM1+X(ISMD-1,J)
           CIM2=CIM2+X(ISMD-2,J)
          END DO
          SXTMP=1.5*CI-2.*CIM1+0.5*CIM2
        END IF
        SYTMP=Y(ISMD,JE)-Y(ISMD,JB)
        SMD=SXTMP/SYTMP
        RETURN
      END IF
C
      IF(JSMD.NE.0) THEN
        SXTMP=X(IE,JSMD)-X(IB,JSMD)
        IF(JSMD.EQ.JB) THEN
          CJ  =0.5*(Y(IB,JSMD  )+Y(IE,JSMD  ))
          CJP1=0.5*(Y(IB,JSMD+1)+Y(IE,JSMD+1))
          CJP2=0.5*(Y(IB,JSMD+2)+Y(IE,JSMD+2))
          DO I=IB+1,IE-1
           CJ  =CJ  +Y(I,JSMD  )
           CJP1=CJP1+Y(I,JSMD+1)
           CJP2=CJP2+Y(I,JSMD+2)
          END DO
          SYTMP=-1.5*CJ+2.*CJP1-0.5*CJP2
        END IF
        IF(JSMD.GT.JB.AND.JSMD.LT.JE) THEN
          CJM1=0.5*(Y(IB,JSMD-1)+Y(IE,JSMD-1))
          CJP1=0.5*(Y(IB,JSMD+1)+Y(IE,JSMD+1))
          DO I=IB+1,IE-1
           CJM1=CJM1+Y(I,JSMD-1)
           CJP1=CJP1+Y(I,JSMD+1)
          END DO
          SYTMP=0.5*(CJP1-CJM1)
        END IF
        IF(JSMD.EQ.JE) THEN
          CJ  =0.5*(Y(IB,JSMD  )+Y(IE,JSMD  ))
          CJM1=0.5*(Y(IB,JSMD-1)+Y(IE,JSMD-1))
          CJM2=0.5*(Y(IB,JSMD-2)+Y(IE,JSMD-2))
          DO I=IB+1,IE-1
           CJ  =CJ  +Y(I,JSMD  )
           CJM1=CJM1+Y(I,JSMD-1)
           CJM2=CJM2+Y(I,JSMD-2)
          END DO
          SYTMP=1.5*CJ-2.*CJM1+0.5*CJM2
        END IF
        SMD=SXTMP/SYTMP
        RETURN
      END IF
C
C********************************************************************C
C
      RETURN
      END
C
C********************************************************************C
C********************************************************************C
C********************************************************************C
C
      REAL*8 FUNCTION XUTMBAY(RLONG,RLAT)
      IMPLICIT REAL*8 (A-H,O-Z)
      REAL*8 RLONG, RLAT
      COMMON /CUTM/  RK,PHI0,CMERID, AE,ECC2,ECC2P
      DATA PII/3.1415926535898/
      DATA DTR/0.017453292519943/
C
C********************************************************************C
C
      X = 500000.
      Y = 0. 
C
      IF (RLAT.GT.80. .OR. RLAT.LT.-80.) THEN
        XUTM=0.
       ELSE
        RN = AE/SQRT(1 - ECC2*(SIN(RLAT*DTR))**2)
        T  = TAN(RLAT*DTR)**2
        C  = ECC2P*COS(RLAT*DTR)**2
        A  = COS(RLAT*DTR)*(rlong - CMERID)*DTR
        XUTM  = RK*RN*(A+(1-T+C)*(A**3)/6.+(5.-18.*T+(T**2)+72*C-58.
     $      *ECC2P)*(A**5)/120.) 
      END IF
      XUTMBAY = X -  XUTM
C
C********************************************************************C
C
      RETURN
      END
C
C********************************************************************C
C********************************************************************C
C********************************************************************C
C      
      REAL*8 FUNCTION YUTMBAY(RLONG,RLAT)
      IMPLICIT REAL*8 (A-H,O-Z)
      REAL*8 RLONG, RLAT
      COMMON /CUTM/  RK,PHI0,CMERID, AE,ECC2,ECC2P
      DATA PII/3.1415926535898/
      DATA DTR/0.017453292519943/
C
C********************************************************************C
C
         X = 500000.
         Y = 0.
C
      RN = AE/SQRT(1 - ECC2*SIN(RLAT*DTR)**2)
      T  = TAN(RLAT*DTR)**2
      C  = ECC2P*COS(RLAT*DTR)**2
      A  = COS(RLAT*DTR)*(RLONG - CMERID)*DTR
      RM =AE*((1-ECC2/4-3*ECC2**2/64-5*ECC2**3/256)*RLAT*DTR-(3*ECC2/8+3
     $*ECC2**2/32+45*ECC2**3/1024)*SIN(2*RLAT*DTR)+(15*ECC2**2/256+45
     $*ECC2**3/1024)*SIN(4*RLAT*DTR)- (35*ECC2**3/3072)*SIN(6*RLAT*DTR))
      RM0=AE*((1-ECC2/4-3*ECC2**2/64-5*ECC2**3/256)*PHI0*DTR-(3*ECC2/8+3
     $*ECC2**2/32+45*ECC2**3/1024)*SIN(2*PHI0*DTR)+(15*ECC2**2/256+45
     $*ECC2**3/1024)*SIN(4*PHI0*DTR)-(35*ECC2**3/3072)*SIN(6*PHI0*DTR))
      IF(RLAT.EQ.90..OR.RLAT.EQ.-90.) THEN
        YUTMBAY = RK*(RM - RM0)
      ELSE
        TEMPA = A**6/720.
        TEMPB = 330.*ECC2P
        TEMPC = 600.*C
        TEMPD = T**2
        TEMPE = 58.*T
        TEMP1 = (61.-TEMPE+TEMPD+TEMPC-TEMPB)*TEMPA
        TEMPA = A**4/24.
        TEMPB = 4*C**2
        TEMPC = 9.*C
        TEMP2 = (5.-T+TEMPC+TEMPB)*TEMPA
        TEMP3=A**2/2
        TEMP = TEMP1+TEMP2+TEMP3
        TEMP = TAN(RLAT*DTR)*TEMP
        TEMP = RN*TEMP
        TEMP = RM+TEMP
        YUTMBAY = RK*TEMP
      END IF
C
C********************************************************************C
C
      RETURN
      END
C
C********************************************************************C
C********************************************************************C
C********************************************************************C
C      
      REAL*8 FUNCTION FIB(YY,J)
      IMPLICIT REAL*8 (A-H,O-Z)
      REAL*8 YY
C
      IF(YY.GE.20.6.AND.YY.LE.35.6) THEN
       FIB=9.*(YY-21.)/15. + 7.
       RETURN
      END IF
C
      WRITE(6,601) YY,J
  601 FORMAT(' FUNCTION FIB OUT OF BOUNDS YY,J = ',F10.4,I8/)
C
C********************************************************************C
C
      RETURN
      END
C
C********************************************************************C
C********************************************************************C
C********************************************************************C
C      
      REAL*8 FUNCTION FIE(YY,J)
      IMPLICIT REAL*8 (A-H,O-Z)
      REAL*8 YY
C
      IF(YY.GE.9.1.AND.YY.LE.23.6) THEN
       FIE=31.
       RETURN
      END IF
C
      WRITE(6,601) YY,J
  601 FORMAT(' FUNCTION FIE OUT OF BOUNDS YY,J = ',F10.4,I8/)
C
C********************************************************************C
C
      RETURN
      END
C
C********************************************************************C
C********************************************************************C
C********************************************************************C
C      
      REAL*8 FUNCTION GJB(XX,I)
      IMPLICIT REAL*8 (A-H,O-Z)
      REAL*8 XX
C
      IF(XX.GE.6.76.AND.XX.LT.8.76) THEN
       X=XX-6.76
       GJB=20.6-0.6*X-(2.254-0.203*X)*X*X/7.7
       RETURN
      END IF
C
      IF(XX.GE.8.76.AND.XX.LT.14.7) THEN
       GJB=-11.2*(XX-7.)/7.7 + 21.
       RETURN
      END IF
C
      IF(XX.GE.14.7.AND.XX.LT.19.4) THEN
C      GJB=-2.1*(XX-14.7)/4.7 + 9.8
       X=XX-14.7
       CTMP=6.764968/(4.7*4.7)
       DTMP=-2.028605/(4.7*4.7*4.7)
       GJB=9.8-11.2*X/7.7+(CTMP+DTMP*X)*X*X
       RETURN
      END IF
C
      IF(XX.GE.19.4.AND.XX.LE.29.0) THEN
       GJB=1.5*(XX-19.4)/11.6 + 7.7
       RETURN
      END IF
C
      IF(XX.GE.29.0.AND.XX.LE.31.) THEN
       X=XX-31.
       GJB=9.1-(0.63+0.085*X)*X*X/11.6
       RETURN
      END IF
C
      WRITE(6,601) XX,I
  601 FORMAT(' FUNCTION GJB OUT OF BOUNDS XX,I = ',F10.4,I8/)
C
C********************************************************************C
C
      RETURN
      END
C
C********************************************************************C
C********************************************************************C
C********************************************************************C
C      
      REAL*8 FUNCTION GJE(XX,I)
      IMPLICIT REAL*8 (A-H,O-Z)
      REAL*8 XX
C
      IF(XX.GE.15.76.AND.XX.LT.17.76) THEN
       X=XX-15.76
       GJE=35.6-0.6*X-(1.696-0.082*X)*X*X/7.5
       RETURN
      END IF
C
      IF(XX.GE.17.76.AND.XX.LT.22.5) THEN
       GJE=-10.3*(XX-16.)/7.5 + 36.
       RETURN
      END IF
C
      IF(XX.GE.22.5.AND.XX.LT.24.5) THEN
       X=XX-22.5
       GJE=(203.05-10.3*X+2.*X*X)/7.5
       RETURN
      END IF
C
      IF(XX.GE.24.5.AND.XX.LT.29.0) THEN
       GJE=-2.3*(XX-23.5)/7.5 + 25.7
       RETURN
      END IF
C
      IF(XX.GE.29.0.AND.XX.LE.31.0) THEN
       X=XX-31.
       GJE=23.6+(1.175+0.2*X)*X*X/7.5
       RETURN
      END IF
C
      WRITE(6,601) XX,I
  601 FORMAT(' FUNCTION GJE OUT OF BOUNDS XX,I = ',F10.4,I8/)
C
C********************************************************************C
C
      RETURN
      END
C
C********************************************************************C
C********************************************************************C
C********************************************************************C
